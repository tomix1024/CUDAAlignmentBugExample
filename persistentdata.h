#pragma once

#include <cstdint>
#include <optional>

namespace opg {

class PersistentData
{
public:
    __device__ PersistentData(uint32_t *data)
    {
        m_data = data;
    }

    template <typename T>
    __forceinline__ __device__ T &at(uint32_t offset)
    {
        return *reinterpret_cast<T*>(m_data + offset);
    }

    uint32_t *m_data;
};

template <typename T>
class PersistentDataEntry
{
public:
    __device__ PersistentDataEntry(PersistentData *pd) :
        m_pd { pd }
    {
        m_value = T(); // default construct
        m_ptr = 0;
    }

    __device__ ~PersistentDataEntry()
    {
        // Avoiding at<T> call avoids the problem.
#ifdef FIX_AVOID_AT_CALL
        *reinterpret_cast<T*>(m_pd->m_data + m_ptr) = m_value;
#else
        m_pd->at<T>(m_ptr) = m_value;
#endif
    }

    __device__ T &value()
    {
        return m_value;
    }

private:
    PersistentData *m_pd;
    // Swapping the member variables avoid the problem.
#ifdef FIX_SWAP_MEMBERS
    T m_value;
    uint32_t m_ptr;
#else
    uint32_t m_ptr;
    T m_value;
#endif
};


} // namespace opg
