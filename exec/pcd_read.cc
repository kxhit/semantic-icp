#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int
main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZL>);

    if (pcl::io::loadPCDFile<pcl::PointXYZL> ("cloudA.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file cloudA.pcd \n");
        return (-1);
    }
    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " data points from test_pcd.pcd with the following fields: "
              << std::endl;
    for (size_t i = 0; i < cloud->points.size (); ++i)
        if (cloud->points[i].label!=0)
         std::cout << "    " << cloud->points[i].x
                   << " "    << cloud->points[i].y
                   << " "    << cloud->points[i].z
                   << " "    << cloud->points[i].label << std::endl;

    return (0);
}
