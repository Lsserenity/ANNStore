#pragma once

namespace annfs::io {

class QueryIoEventListener {
public:
    virtual ~QueryIoEventListener() = default;

    virtual void OnQueryIoSubmitted(int query_id, int count) = 0;
    virtual void OnQueryIoCompleted(int query_id, int count) = 0;
};

} // namespace annfs::io
