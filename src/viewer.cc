#include "iridescence_viewer/viewer.h"

#include "stella_vslam/config.h"
#include "stella_vslam/system.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/publish/frame_publisher.h"
#include "stella_vslam/publish/map_publisher.h"
#include "stella_vslam/util/yaml.h"

#include <glk/primitives/primitives.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/texture_opencv.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <chrono>
#include <thread>

namespace {
int parse_int(const std::string& msg) {
    int ret = -1;
    try {
        ret = stoi(msg);
    }
    catch (std::invalid_argument& e) {
    }
    catch (std::out_of_range& e) {
    }
    return ret;
}

Eigen::Matrix4d rotate_pose(Eigen::Matrix4d pose_cw, Eigen::Matrix3d rot) {
    return (rot * Eigen::Affine3d(Eigen::Translation3d(pose_cw.block<3, 1>(0, 3)) * pose_cw.block<3, 3>(0, 0))).matrix();
}
} // namespace

namespace iridescence_viewer {

viewer::viewer(const YAML::Node& yaml_node,
               const std::shared_ptr<stella_vslam::system>& system,
               const std::shared_ptr<stella_vslam::publish::frame_publisher>& frame_publisher,
               const std::shared_ptr<stella_vslam::publish::map_publisher>& map_publisher)
    : system_(system), frame_publisher_(frame_publisher), map_publisher_(map_publisher),
      interval_ms_(1000.0f / yaml_node["fps"].as<float>(30.0)) {}

void viewer::run() {
    auto viewer = guik::LightViewer::instance();
    float current_frame_scale = 0.05f;
    float keyframe_scale = 0.05f;
    viewer->register_ui_callback("ui", [&]() {
        ImGui::DragFloat("CurrentFrame scale", &current_frame_scale, 0.01f, 0.01f, 1.0f);
        ImGui::DragFloat("Keyframe scale", &keyframe_scale, 0.01f, 0.01f, 1.0f);
        if (ImGui::Button("Close")) {
            viewer->close();
        }
    });

    while (viewer->spin_once()) {
        viewer->clear_drawables();

        Eigen::Matrix3d rot_ros_to_cv_map_frame = (Eigen::Matrix3d() << 0, 0, 1,
                                                   -1, 0, 0,
                                                   0, -1, 0)
                                                      .finished();

        std::vector<std::shared_ptr<stella_vslam::data::keyframe>> keyfrms;
        map_publisher_->get_keyframes(keyfrms);

        std::vector<std::shared_ptr<stella_vslam::data::landmark>> landmarks;
        std::set<std::shared_ptr<stella_vslam::data::landmark>> local_landmarks;
        map_publisher_->get_landmarks(landmarks, local_landmarks);

        Eigen::Matrix4d current_frame_pose = rotate_pose(map_publisher_->get_current_cam_pose().inverse().eval(), rot_ros_to_cv_map_frame);
        viewer->update_drawable("current_frame", glk::Primitives::wire_frustum(), guik::FlatColor(Eigen::Vector4f(0.7f, 0.7f, 1.0f, 1.0f), current_frame_pose).scale(current_frame_scale));

        for (const auto& keyfrm : keyfrms) {
            if (!keyfrm || keyfrm->will_be_erased()) {
                continue;
            }
            Eigen::Matrix4d keyfrms_pose = rotate_pose(keyfrm->get_pose_wc(), rot_ros_to_cv_map_frame);
            const auto name = std::string("keyfrms_pose_") + std::to_string(keyfrm->id_);
            viewer->update_drawable(name, glk::Primitives::wire_frustum(), guik::FlatColor(Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f), keyfrms_pose).scale(keyframe_scale));
        }

        std::vector<Eigen::Vector3f> points;
        for (const auto& lm : landmarks) {
            if (!lm || lm->will_be_erased()) {
                continue;
            }
            const stella_vslam::Vec3_t pos_w = rot_ros_to_cv_map_frame * lm->get_pos_in_world();
            points.push_back(pos_w.cast<float>());
        }
        auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(points);
        viewer->update_drawable("map", cloud_buffer, guik::FlatColor(Eigen::Vector4f(1.0f, 0.5f, 0.0f, 1.0f)));

        viewer->set_draw_xy_grid(false);
        viewer->update_drawable("coordinate_system", glk::Primitives::coordinate_system(), guik::VertexColor());

        cv::Mat frame = frame_publisher_->draw_frame();
        viewer->update_image("frame", glk::create_texture(frame));

        if (terminate_is_requested()) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
    }

    if (system_->tracker_is_paused()) {
        system_->resume_tracker();
    }

    system_->request_terminate();

    terminate();
}

void viewer::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}

bool viewer::is_terminated() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return is_terminated_;
}

bool viewer::terminate_is_requested() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void viewer::terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    is_terminated_ = true;
}

} // namespace iridescence_viewer
