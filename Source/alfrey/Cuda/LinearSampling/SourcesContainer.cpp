#include "SourcesContainer.h"

bool SourcesContainer::is_new_source(const source_type_ptr& ptr) {
    bool not_a_source = false;
    for (auto iter = _sources.begin(); iter != _sources.end(); ++iter) {
        if (Preq(*iter->second, *ptr)) {
            return false;
        }
    }

    return true;
}

void SourcesContainer::remove_old_sources(const source_type_ptr& new_source_ptr) {
    _sources.erase(
        std::remove_if(
            _sources.begin(), 
            _sources.end(), 
            [&](container_elem_type a) 
            {
                return Preq(*new_source_ptr, *a.second);
            }
        ),
    _sources.end());
}
