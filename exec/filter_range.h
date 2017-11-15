#ifndef _FILTER_RANGE_H_
#define _FILTER_RANGE_H_



void filterRange(pcl::PointCloud<pcl::PointXYZL>::Ptr cloudPtr,
                 const double range) {

    auto it = cloudPtr->begin();
    while( it != cloudPtr->end()) {
        pcl::PointXYZL pt = *it;
        if((pt.x*pt.x+pt.y*pt.y+pt.z*pt.z)>range*range) {
            cloudPtr->erase(it);
        } else {
            it++;
        }
    }
}

#endif // #ifndef _FILTER_RANGE_H_
