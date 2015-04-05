#ifndef __SOURCES_CONTAINER_H
#define __SOURCES_CONTAINER_H

#include <algorithm>
#include <list>
#include <memory>
#include <utility>
#include <vector>

// Set of support methods to maintain the list of "sources" (classifiers which have no precesor according to Preq relation (see below).
class SourcesContainer {
public:
    typedef std::vector<unsigned char> source_type;                 // type to represent error vector of a classifiers
    typedef std::shared_ptr<source_type> source_type_ptr;           // shared pointer to source_type
    typedef std::pair<int, source_type_ptr> container_elem_type;    // container stores a user-defined key associated with the classifier
    typedef std::list<container_elem_type> container_type;          // container type

    // Creates source_type_ptr from raw memory
    static source_type_ptr ConstructSourcePtr(const unsigned char* begin, const unsigned char* end) {
        return source_type_ptr(new SourcesContainer::source_type(begin, end));
    }

    // Tests whether lhs vector "preceeds" rhs vector
    // Note: this is NOT comparison in lexicographic order. 
    // - If lhs[i] < rhs[i] then lex order says "smaller" regardles of lhs[j] vs rhs[j] for j > i.
    // - "Preq" require lhs[i] <= rhs[i] for all i.
    static bool Preq(const source_type& lhs, const source_type& rhs) {
        for (auto lhsIter = lhs.begin(), rhsIter = rhs.begin(); lhsIter != lhs.end(); ++lhsIter, ++rhsIter) {
            if (*lhsIter > *rhsIter) return false;
        }

        return true;
    }

    // Checks wehther ptr is a new source.
    // Require sequential scan over the container.
    bool is_new_source(const source_type_ptr& ptr);

    // Removes all sources that Preq to new_source_ptr
    void remove_old_sources(const source_type_ptr& new_source_ptr);

    // Adds new element to container.
    void push_back(int id, const source_type_ptr& new_source_ptr) {
        _sources.push_back(container_elem_type(id, new_source_ptr));
    }

    size_t size() {
        return _sources.size();
    }

    container_type::iterator begin() {
        return _sources.begin();
    }

    container_type::iterator end() {
        return _sources.end();
    }
private:
    container_type _sources;
};


#endif //__SOURCES_CONTAINER_H

