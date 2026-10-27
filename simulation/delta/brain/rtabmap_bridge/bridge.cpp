// RTAB-Map behind a pipe: stereo frames in on stdin, poses out on stdout.
//
// This is "Estimate wearer position" on the slide. RTAB-Map is C++ and has no
// Python bindings, so instead of binding it, the brain (spatial_agent/brain/
// localize.py) starts this program and talks to it over its standard streams.
// That keeps RTAB-Map's own dependencies (OpenCV 5, PCL, g2o, GTSAM) in their
// own conda env, well away from torch.
//
// Two RTAB-Map pieces do the work, the same split its ROS nodes use:
//
//     Odometry   frame to frame: where the glasses moved since the last frame.
//                Runs on EVERY frame. Drifts slowly.
//     Rtabmap    the map: remembers keyframes as nodes, notices when the wearer
//                is back somewhere already seen (a loop closure), and corrects
//                the drift. Runs only on frames the brain marks as keyframes.
//
// A pose out is   map_from_base = correction (map_from_odom) * odometry.
// "base" is the left camera with ROS axes: x forward, y left, z up.
//
// Wire format, little-endian, one message after another on stdin:
//
//     "FRAM" int32 id, float64 stamp, int32 width, int32 height, int32 flags,
//            float64 imu yaw, float64 imu pitch, float64 imu roll,
//            float64 odometer, then
//            width*height bytes of left grey, then the same of right.
//            flags bit 0: keyframe -- also add this frame to the map.
//            flags bit 1: the imu yaw is real (radians, anticlockwise).
//            flags bit 2: pitch (+ looks down) and roll (+ tilts right), in
//            radians from gravity, are real: the camera is mounted tilted
//            on a level body for this frame.
//            flags bit 3: the odometer is real -- metres rolled so far, a
//            running total from the cage's axle encoder. Not set in the air.
//     "QUIT" save the database (if any) and exit.
//
// Every reply is one line on stdout starting with "@@ " and holding JSON.
// Anything else on stdout is a library talking, and the brain ignores it.

#include <rtabmap/core/IMU.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/StereoCameraModel.h>
#include <rtabmap/core/Version.h>
#include <rtabmap/utilite/ULogger.h>

#include <opencv2/core.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace rtabmap;

namespace {

// After this many frames in a row without an odometry fix, carry on from the
// last good pose instead of waiting to see the same features again. Resetting
// to the last pose keeps one continuous frame; RTAB-Map's own reset would
// start a fresh map at the origin and leave the brain two disconnected maps.
const int LOST_FRAMES_BEFORE_RESET = 3;

// While blind, the wearer is taken to keep walking as they were: the speed
// of the last tracked moments, in the direction the IMU says they face --
// for at most GUESS_SECONDS, and not while the IMU shows them turning on the
// spot. Assuming they stopped left the map that much behind them: a blind
// half-second at walking pace became a lasting half-metre offset, and with
// frames skipped while RTAB-Map was busy, up to a metre and a half.
const double GUESS_SECONDS = 1.5;
const double TURNING_IN_PLACE = 0.5;   // rad/s of IMU turn: no walking guessed
const double VELOCITY_BLEND = 0.3;     // weight of each tracked step in the speed

// How unsure to be about motion during a lost frame: whatever the wearer did
// then went unseen. 0.05 m^2 and 0.05 rad^2 are a 22 cm and 13 degree
// standard deviation per frame -- loose enough that a later loop closure is
// free to straighten it out.
const double LOST_LINEAR_VARIANCE = 0.05;
const double LOST_ANGULAR_VARIANCE = 0.05;

// The IMU's heading, trusted to about half a degree as an orientation, and
// to a fifth of a degree for the turn between two frames.
const double IMU_VARIANCE = 1e-4;
const double IMU_STEP_VARIANCE = 1.2e-5;

// When to take the turn from the IMU instead of the cameras: vision tracked
// fewer features than this (good frames track 100 to 300; the frames that
// went wrong in front of blank walls tracked 25 to 80), or it disagrees with
// the gyro by more than this many degrees in one step.
const int WEAK_INLIERS = 80;
const double DISAGREE_DEGREES = 1.0;

// Used when a registration reports no covariance of its own.
const double DEFAULT_VARIANCE = 1e-4;

// A loop closure has to agree with odometry. Houses repeat themselves -- two
// doorways, two runs of kitchen cupboards -- and RTAB-Map will occasionally
// "recognise" a place it has never been. Its own check looks at one link at
// a time, and a false closure spreading a 140 degree bend over fifteen links
// passed it at under 2 sigma each. So: since the last trusted closure, how
// far could odometry have drifted? A closure that moves the wearer further
// than that, or turns them more, is a lookalike, and is taken back out.
const double ALLOWED_JUMP = 0.5;              // metres, always

// These two are settings (--allowed-jump-per-metre, --allowed-jump-per-lost)
// because they describe the robot, not RTAB-Map. The defaults are the
// glasses': a person drifts more per metre walked, and covers up to a quarter
// of a metre in a frame nobody saw. A small drone does neither, and with
// these values a 3.5 m lookalike closure passed the guard after 18 lost
// frames (allowed 6.6 m) and threw the whole map 4.8 m off.
double ALLOWED_JUMP_PER_METRE = 0.05;         // per metre travelled
double ALLOWED_JUMP_PER_LOST = 0.25;          // per lost frame
const double ALLOWED_TURN = 10.0;             // degrees, always
const double ALLOWED_TURN_PER_DEGREE = 0.05;  // per degree turned
const double ALLOWED_TURN_PER_LOST = 10.0;    // per lost frame, cameras only
const double ALLOWED_TURN_PER_LOST_WITH_IMU = 1.0;

bool readExact(void * buffer, size_t size)
{
    return std::fread(buffer, 1, size, stdin) == size;
}

std::string matrix(const Transform & transform)
{
    if (transform.isNull())
    {
        return "null";
    }

    // Row-major 3x4: the rotation and translation, what Transform stores.
    std::ostringstream out;
    out.precision(7);
    out << "[";

    const float * values = transform.data();

    for (int i = 0; i < 12; ++i)
    {
        out << (i ? "," : "") << values[i];
    }

    out << "]";
    return out.str();
}

void emit(const std::string & json)
{
    std::fprintf(stdout, "@@ %s\n", json.c_str());
    std::fflush(stdout);
}

std::string quoted(const std::string & text)
{
    std::string out = "\"";

    for (char c : text)
    {
        if (c == '"' || c == '\\')
        {
            out += '\\';
        }

        out += (c == '\n') ? ' ' : c;
    }

    return out + "\"";
}

float yawOf(const Transform & transform)
{
    float roll, pitch, yaw;
    transform.getEulerAngles(roll, pitch, yaw);

    return yaw;
}

double degreesBetween(float a, float b)
{
    return std::fabs(std::remainder(a - b, 2.0 * M_PI)) * 180.0 / M_PI;
}

bool sameTransform(const Transform & a, const Transform & b)
{
    if (a.isNull() || b.isNull())
    {
        return a.isNull() == b.isNull();
    }

    for (int i = 0; i < 12; ++i)
    {
        if (std::fabs(a.data()[i] - b.data()[i]) > 1e-3f)
        {
            return false;
        }
    }

    return true;
}

// The map each node belongs to. A database reopened in a later run starts a
// new map; when RTAB-Map recognises a place from the earlier run, the two
// maps join, and the oldest node of the joined graph belongs to map 0.
int mapOf(const Rtabmap & slam, int node)
{
    if (node <= 0 || !slam.getMemory())
    {
        return -1;
    }

    return slam.getMemory()->getMapId(node, true);
}

}  // namespace


int main(int argc, char ** argv)
{
    double fx = 0, fy = 0, cx = 0, cy = 0, baseline = 0;
    int width = 0, height = 0;
    std::string database;
    std::string logFile;
    ParametersMap parameters;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "%s needs a value\n", arg.c_str());
                std::exit(2);
            }

            return argv[++i];
        };

        if (arg == "--version")
        {
            std::printf("rtabmap_bridge, RTAB-Map %s\n", RTABMAP_VERSION);
            return 0;
        }
        else if (arg == "--fx") fx = std::stod(next());
        else if (arg == "--fy") fy = std::stod(next());
        else if (arg == "--cx") cx = std::stod(next());
        else if (arg == "--cy") cy = std::stod(next());
        else if (arg == "--baseline") baseline = std::stod(next());
        else if (arg == "--width") width = std::stoi(next());
        else if (arg == "--height") height = std::stoi(next());
        else if (arg == "--database") database = next();
        else if (arg == "--allowed-jump-per-metre") ALLOWED_JUMP_PER_METRE = std::stod(next());
        else if (arg == "--allowed-jump-per-lost") ALLOWED_JUMP_PER_LOST = std::stod(next());
        else if (arg == "--log") logFile = next();
        else if (arg == "--param")
        {
            // Key=Value, with the key checked against RTAB-Map's own list so a
            // typo fails loudly instead of being silently ignored.
            std::string pair = next();
            size_t equals = pair.find('=');
            std::string key = pair.substr(0, equals);

            if (equals == std::string::npos
                || Parameters::getDefaultParameters().count(key) == 0)
            {
                std::fprintf(stderr, "Unknown RTAB-Map parameter: %s\n", pair.c_str());
                return 2;
            }

            parameters[key] = pair.substr(equals + 1);
        }
        else
        {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    if (fx <= 0 || fy <= 0 || baseline <= 0 || width <= 0 || height <= 0)
    {
        std::fprintf(stderr, "Need --fx --fy --cx --cy --baseline --width --height\n");
        return 2;
    }

    // RTAB-Map's console logger writes to stdout, which is our channel.
    if (logFile.empty())
    {
        ULogger::setType(ULogger::kTypeNoLog);
    }
    else
    {
        ULogger::setType(ULogger::kTypeFile, logFile, false);
        ULogger::setLevel(getenv("RTABMAP_BRIDGE_DEBUG") ? ULogger::kInfo : ULogger::kWarning);
    }

    // Rectified, parallel stereo: one focal length, one principal point, and
    // the baseline. The local transform turns the optical frame (z forward)
    // into the base frame (x forward), which is RTAB-Map's default.
    StereoCameraModel model(
        "glasses",
        fx, fy, cx, cy, baseline,
        Transform(0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0),
        cv::Size(width, height));

    std::unique_ptr<Odometry> odometry(Odometry::create(parameters));

    Rtabmap slam;
    slam.init(parameters, database);

    size_t remembered = slam.getMemory()
        ? slam.getMemory()->getAllSignatureIds().size()
        : 0;

    emit(std::string("{\"type\":\"ready\",\"version\":") + quoted(RTABMAP_VERSION)
         + ",\"nodes_loaded\":" + std::to_string(remembered) + "}");

    Transform lastCorrection = slam.getMapCorrection();

    // Two odometries. `vision` is RTAB-Map's own, from the cameras alone.
    // `odom` is what the map is given: the same motion, except that the
    // heading comes from the IMU whenever the cameras are unsure of it. Each
    // is anchored at the last frame vision tracked, with the IMU heading of
    // that moment, so a gap of lost frames is bridged in one step.
    Transform visionAnchor;
    Transform odomAnchor = Transform::getIdentity();
    double imuAnchor = 0.0;
    bool imuAnchored = false;

    Transform odom = Transform::getIdentity();
    Transform previousOdom = odom;
    int lostFrames = 0;

    // Walking speed in the body frame (m/s), from the tracked steps.
    double velocityX = 0.0, velocityY = 0.0;
    double lastTrackedStamp = -1.0;

    // The odometer's reading at the last frame vision tracked. While blind on
    // the ground, how far the cage has rolled since then is MEASURED, so the
    // guess above is not needed: a drone pressed against a wall reads zero
    // and stays put, where the guess drove it on at its last speed for 1.5 s.
    double odometerAnchor = 0.0;
    bool odometerAnchored = false;

    // How uncertain the odometry is since the last keyframe went to the map.
    // Uncertainties of successive steps add. This is what lets the map tell
    // a real loop closure from a lookalike: a closure that would bend
    // confident odometry by 140 degrees costs too much and is rejected. Left
    // at RTAB-Map's default (identity: a 57 degree standard deviation), any
    // closure at all looked plausible.
    cv::Mat covariance = cv::Mat::zeros(6, 6, CV_64FC1);

    // Odometry since the last trusted loop closure, for checking the next.
    double travelled = 0.0;
    double turned = 0.0;
    int lostSinceClosure = 0;
    int lastRootMap = -1;

    std::vector<unsigned char> left, right;

    while (true)
    {
        char magic[4];

        if (!readExact(magic, 4) || std::memcmp(magic, "QUIT", 4) == 0)
        {
            break;
        }

        if (std::memcmp(magic, "FRAM", 4) != 0)
        {
            emit("{\"type\":\"error\",\"message\":\"bad magic\"}");
            break;
        }

        int32_t id, frameWidth, frameHeight, flags;
        double stamp, imuYaw, imuPitch, imuRoll, odometer;

        if (!readExact(&id, 4) || !readExact(&stamp, 8)
            || !readExact(&frameWidth, 4) || !readExact(&frameHeight, 4)
            || !readExact(&flags, 4) || !readExact(&imuYaw, 8)
            || !readExact(&imuPitch, 8) || !readExact(&imuRoll, 8)
            || !readExact(&odometer, 8))
        {
            break;
        }

        bool hasImu = (flags & 2) != 0;
        bool hasOdometer = (flags & 8) != 0;

        size_t pixels = static_cast<size_t>(frameWidth) * frameHeight;
        left.resize(pixels);
        right.resize(pixels);

        if (!readExact(left.data(), pixels) || !readExact(right.data(), pixels))
        {
            break;
        }

        auto started = std::chrono::steady_clock::now();

        // Cloned, because SensorData keeps the matrices and the buffers above
        // are reused for the next frame.
        // A tilted head: the body ("base") stays level -- which is what 3-DoF
        // mapping assumes -- and the camera sits on it pitched and rolled by
        // what the IMU measures from gravity. Without this, looking down is
        // read as the whole body tipping, and the 3-DoF projection of that
        // throws the position off by metres.
        StereoCameraModel frameModel = model;

        if (flags & 4)
        {
            Transform tilt(0, 0, 0, static_cast<float>(imuRoll), static_cast<float>(imuPitch), 0);
            frameModel.setLocalTransform(tilt * model.localTransform());
        }

        SensorData data(
            cv::Mat(frameHeight, frameWidth, CV_8UC1, left.data()).clone(),
            cv::Mat(frameHeight, frameWidth, CV_8UC1, right.data()).clone(),
            frameModel,
            0,
            stamp);

        // The IMU's heading, as an orientation about the vertical. RTAB-Map's
        // odometry then takes its rotation guess from the IMU instead of
        // assuming the last motion continues -- which is what keeps a turn
        // in front of a blank wall from being read as no turn at all.
        if (hasImu)
        {
            cv::Mat variance = cv::Mat::eye(3, 3, CV_64FC1) * IMU_VARIANCE;

            data.setIMU(IMU(
                cv::Vec4d(0.0, 0.0, std::sin(imuYaw / 2.0), std::cos(imuYaw / 2.0)),
                variance,
                cv::Vec3d(0.0, 0.0, 0.0), variance,
                cv::Vec3d(0.0, 0.0, 9.81), variance,
                Transform::getIdentity()));
        }

        OdometryInfo info;
        Transform vision = odometry->process(data, &info);
        bool ok = !vision.isNull();

        // How far the IMU says the head turned since vision last tracked.
        bool imuGap = hasImu && imuAnchored;
        double imuTurn = imuGap ? std::remainder(imuYaw - imuAnchor, 2.0 * M_PI) : 0.0;

        cv::Mat stepVariance = cv::Mat::zeros(6, 6, CV_64FC1);
        bool headingFromImu = false;

        if (ok)
        {
            lostFrames = 0;

            Transform step = visionAnchor.isNull()
                ? Transform::getIdentity()
                : visionAnchor.inverse() * vision;

            // The registration's own uncertainty. The first frame after a
            // reset reports 9999: "no idea". Taken literally, the link across
            // the gap costs nothing to bend, and a lookalike loop closure once
            // rotated half the map by 145 degrees through exactly such a link.
            // What is actually unknown is one lost frame's worth of motion.
            const cv::Mat & reported = info.reg.covariance;
            bool hasCovariance = reported.rows == 6 && reported.cols == 6
                && reported.type() == CV_64FC1;

            for (int i = 0; i < 6; ++i)
            {
                double variance = hasCovariance ? reported.at<double>(i, i) : DEFAULT_VARIANCE;

                if (variance >= 1.0)
                {
                    variance = i < 3 ? LOST_LINEAR_VARIANCE : LOST_ANGULAR_VARIANCE;
                }

                stepVariance.at<double>(i, i) = variance;
            }

            // Vision is unsure of the turn when it tracked few features, or
            // when it disagrees with the gyro -- which, over a tenth of a
            // second, is far more trustworthy than a handful of features on a
            // blank wall. Then the gyro's turn wins and vision keeps the
            // walking part.
            if (imuGap)
            {
                bool weak = info.reg.inliers < WEAK_INLIERS;
                bool disagree = degreesBetween(yawOf(step), imuTurn) > DISAGREE_DEGREES;

                if (weak || disagree)
                {
                    step = Transform(step.x(), step.y(), 0.0f, 0.0f, 0.0f, imuTurn);
                    headingFromImu = true;

                    for (int i = 3; i < 6; ++i)
                    {
                        stepVariance.at<double>(i, i) = IMU_STEP_VARIANCE;
                    }
                }
            }

            odom = odomAnchor * step;

            double since = lastTrackedStamp < 0.0 ? 0.0 : stamp - lastTrackedStamp;

            if (since > 0.02 && since < 0.6)
            {
                velocityX = (1.0 - VELOCITY_BLEND) * velocityX + VELOCITY_BLEND * step.x() / since;
                velocityY = (1.0 - VELOCITY_BLEND) * velocityY + VELOCITY_BLEND * step.y() / since;
            }

            lastTrackedStamp = stamp;
            visionAnchor = vision;
            odomAnchor = odom;
            odometerAnchor = odometer;
            odometerAnchored = hasOdometer;
        }
        else
        {
            ++lostSinceClosure;

            Transform guess;

            if (hasOdometer && odometerAnchored)
            {
                // Blind on the ground: the turning is the IMU's and the
                // rolling is the encoder's. The distance is laid along the
                // heading halfway through the turn -- the chord of the arc a
                // drone rolling and turning at once traces.
                double rolled = odometer - odometerAnchor;
                guess = Transform(static_cast<float>(rolled * std::cos(imuTurn / 2.0)),
                                  static_cast<float>(rolled * std::sin(imuTurn / 2.0)), 0.0f,
                                  0.0f, 0.0f, static_cast<float>(imuTurn));
            }
            else
            {
                // Blind in the air, or with no encoder: the turning is the
                // IMU's, and the walking goes on as it was (see
                // GUESS_SECONDS) -- unless the IMU shows a turn on the spot.
                double blind = lastTrackedStamp < 0.0 ? 0.0 : std::min(stamp - lastTrackedStamp, GUESS_SECONDS);
                bool turningInPlace = blind > 0.0 && std::fabs(imuTurn) / blind > TURNING_IN_PLACE;
                double walk = turningInPlace ? 0.0 : blind;
                guess = Transform(static_cast<float>(velocityX * walk), static_cast<float>(velocityY * walk), 0.0f,
                                  0.0f, 0.0f, static_cast<float>(imuTurn));
            }

            odom = odomAnchor * guess;
            headingFromImu = imuGap;

            for (int i = 0; i < 6; ++i)
            {
                stepVariance.at<double>(i, i) = i < 3
                    ? LOST_LINEAR_VARIANCE
                    : (imuGap ? IMU_STEP_VARIANCE : LOST_ANGULAR_VARIANCE);
            }

            // Still blind after a few frames: restart visual odometry where it
            // guesses the wearer is, turned as far as the IMU says.
            if (++lostFrames >= LOST_FRAMES_BEFORE_RESET && !visionAnchor.isNull())
            {
                // ...from where the wearer has walked to by now, not where
                // vision last saw them.
                visionAnchor = visionAnchor * guess;
                odomAnchor = odomAnchor * guess;
                odometry->reset(visionAnchor);
                lostFrames = 0;
                lastTrackedStamp = stamp;

                // The anchors have moved to here, so the rolling since is
                // counted from here.
                odometerAnchor = odometer;
                odometerAnchored = hasOdometer;

                if (imuGap)
                {
                    imuAnchor = imuYaw;
                }
            }
        }

        if (hasImu && (ok || !imuAnchored))
        {
            imuAnchor = imuYaw;
            imuAnchored = true;
        }

        covariance += stepVariance;

        Transform moved = previousOdom.inverse() * odom;
        travelled += moved.getNorm();
        turned += degreesBetween(yawOf(odom), yawOf(previousOdom));
        previousOdom = odom;

        int node = 0;
        int loop = 0;
        int rejected = 0;
        double jump = 0.0;
        double turn = 0.0;
        bool added = false;

        if (ok && (flags & 1))
        {
            added = slam.process(data, odom, covariance);
            loop = slam.getLoopClosureId();
            node = slam.getLastLocationId();

            // Only a new node starts a new stretch of odometry. A rejected
            // keyframe (too little movement) leaves the stretch running.
            if (added)
            {
                covariance = cv::Mat::zeros(6, 6, CV_64FC1);
            }

            if (loop > 0)
            {
                Transform before = lastCorrection * odom;
                Transform after = slam.getMapCorrection() * odom;

                jump = before.getDistance(after);
                turn = degreesBetween(yawOf(after), yawOf(before));

                // Recognising last session's map is supposed to jump: the two
                // sessions started in unrelated frames. Only closures within
                // one connected map are held to odometry.
                bool joinsEarlierSession =
                    mapOf(slam, loop) != mapOf(slam, node) && lastRootMap == mapOf(slam, node);

                double allowedJump = ALLOWED_JUMP
                    + ALLOWED_JUMP_PER_METRE * travelled
                    + ALLOWED_JUMP_PER_LOST * lostSinceClosure;

                double allowedTurn = ALLOWED_TURN
                    + ALLOWED_TURN_PER_DEGREE * turned
                    + (hasImu ? ALLOWED_TURN_PER_LOST_WITH_IMU : ALLOWED_TURN_PER_LOST)
                        * lostSinceClosure;

                if (!joinsEarlierSession && (jump > allowedJump || turn > allowedTurn))
                {
                    slam.rejectLastLoopClosure();
                    rejected = loop;
                    loop = 0;
                }
                else
                {
                    travelled = 0.0;
                    turned = 0.0;
                    lostSinceClosure = 0;
                }
            }
        }

        Transform correction = slam.getMapCorrection();
        Transform pose = ok ? correction * odom : Transform();

        // The node this keyframe hangs off, and where the map currently puts
        // it. If a later loop closure moves the node, the brain moves every
        // keyframe hanging off it by the same amount.
        Transform nodePose;
        const std::map<int, Transform> & graph = slam.getLocalOptimizedPoses();

        if (node > 0)
        {
            auto found = graph.find(node);

            if (found != graph.end())
            {
                nodePose = found->second;
            }
        }

        int rootMap = graph.empty() ? -1 : mapOf(slam, graph.begin()->first);

        if (rootMap >= 0)
        {
            lastRootMap = rootMap;
        }

        double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();

        std::ostringstream reply;
        reply << "{\"type\":\"pose\",\"id\":" << id
              << ",\"ok\":" << (ok ? "true" : "false")
              << ",\"pose\":" << matrix(pose)
              << ",\"odom\":" << matrix(odom)
              << ",\"imu_heading\":" << (headingFromImu ? "true" : "false")
              << ",\"inliers\":" << info.reg.inliers
              << ",\"matches\":" << info.reg.matches
              << ",\"features\":" << info.features
              << ",\"keyframe\":" << ((flags & 1) ? "true" : "false")
              << ",\"added\":" << (added ? "true" : "false")
              << ",\"node\":" << node
              << ",\"node_pose\":" << matrix(nodePose)
              << ",\"loop\":" << loop
              << ",\"rejected\":" << rejected
              << ",\"jump\":" << jump
              << ",\"turn\":" << turn
              << ",\"map\":" << mapOf(slam, node)
              << ",\"root_map\":" << rootMap
              << ",\"ms\":" << elapsed
              << "}";

        emit(reply.str());

        // The correction moved, so the whole graph may have moved: send every
        // node's pose so the brain can redraw its map from its keyframes.
        if (!sameTransform(correction, lastCorrection) || loop > 0)
        {
            std::ostringstream poses;
            poses << "{\"type\":\"graph\",\"poses\":{";

            bool first = true;

            for (const auto & entry : graph)
            {
                poses << (first ? "" : ",") << "\"" << entry.first << "\":"
                      << matrix(entry.second);
                first = false;
            }

            poses << "}}";
            emit(poses.str());
        }

        lastCorrection = correction;
    }

    slam.close(!database.empty());
    emit("{\"type\":\"closed\"}");

    return 0;
}
