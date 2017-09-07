file (GLOB HEADER_FILES *.h *.hpp)

add_executable (pcd_read pcd_read.cc)
target_link_libraries (pcd_read ${PCL_LIBRARIES})

add_executable (pcd_write pcd_write.cc)
target_link_libraries (pcd_write ${PCL_LIBRARIES})

add_executable (make_semantic make_semantic.cc)
target_link_libraries (make_semantic semantic_icp_lib ${PCL_LIBRARIES})

add_executable (test_icp test_icp.cc)
target_link_libraries (test_icp semantic_icp_lib ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable (kitti_eval kitti_eval.cc ${HEADER_FILES})
target_link_libraries (kitti_eval semantic_icp_lib ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable (roc_eval roc_eval.cc ${HEADER_FILES})
target_link_libraries (roc_eval semantic_icp_lib ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable (nyu_eval nyu_eval.cc ${HEADER_FILES})
target_link_libraries (nyu_eval semantic_icp_lib ${PCL_LIBRARIES} ${CERES_LIBRARIES})
