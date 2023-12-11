#ifndef IRIDESCENCE_VIEWER_VIEWER_H
#define IRIDESCENCE_VIEWER_VIEWER_H

#include "stella_vslam/type.h"
#include "stella_vslam/util/yaml.h"

#include <memory>
#include <mutex>

namespace stella_vslam {

class config;
namespace data {
class keyframe;
class landmark;
} // namespace data

class system;

namespace publish {
class frame_publisher;
class map_publisher;
} // namespace publish

} // namespace stella_vslam

namespace iridescence_viewer {

class viewer {
public:
    /**
     * Constructor
     * @param yaml_node
     * @param system
     * @param frame_publisher
     * @param map_publisher
     */
    viewer(const YAML::Node& yaml_node,
           const std::shared_ptr<stella_vslam::system>& system,
           const std::shared_ptr<stella_vslam::publish::frame_publisher>& frame_publisher,
           const std::shared_ptr<stella_vslam::publish::map_publisher>& map_publisher);

    /**
     * Main loop for window refresh
     */
    void run();

    /**
     * Request to terminate the viewer
     * (NOTE: this function does not wait for terminate)
     */
    void request_terminate();

    /**
     * Check if the viewer is terminated or not
     * @return whether the viewer is terminated or not
     */
    bool is_terminated();

private:
    //! system
    const std::shared_ptr<stella_vslam::system> system_;
    //! frame publisher
    const std::shared_ptr<stella_vslam::publish::frame_publisher> frame_publisher_;
    //! map publisher
    const std::shared_ptr<stella_vslam::publish::map_publisher> map_publisher_;

    const unsigned int interval_ms_;

    //-----------------------------------------
    // management for terminate process

    //! mutex for access to terminate procedure
    mutable std::mutex mtx_terminate_;

    /**
     * Check if termination is requested or not
     * @return
     */
    bool terminate_is_requested();

    /**
     * Raise the flag which indicates the main loop has been already terminated
     */
    void terminate();

    //! flag which indicates termination is requested or not
    bool terminate_is_requested_ = false;
    //! flag which indicates whether the main loop is terminated or not
    bool is_terminated_ = true;
};

} // namespace iridescence_viewer

#endif // IRIDESCENCE_VIEWER_VIEWER_H
