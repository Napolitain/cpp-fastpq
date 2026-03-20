#include <cstdlib>

#include <folly/ScopeGuard.h>
#include <folly/memory/detail/MallocImpl.h>

extern "C" {

#if !FOLLY_HAVE_WEAK_SYMBOLS && !defined(_MSC_VER)
#if (!defined(USE_JEMALLOC) && !defined(FOLLY_USE_JEMALLOC)) || FOLLY_SANITIZE
void* (*mallocx)(size_t, int) = nullptr;
void* (*rallocx)(void*, size_t, int) = nullptr;
size_t (*xallocx)(void*, size_t, size_t, int) = nullptr;
size_t (*sallocx)(const void*, int) = nullptr;
void (*dallocx)(void*, int) = nullptr;
void (*sdallocx)(void*, size_t, int) = nullptr;
size_t (*nallocx)(size_t, int) = nullptr;
int (*mallctl)(const char*, void*, size_t*, void*, size_t) = nullptr;
int (*mallctlnametomib)(const char*, size_t*, size_t*) = nullptr;
int (*mallctlbymib)(const size_t*, size_t, void*, size_t*, void*, size_t) = nullptr;
#endif
bool (*MallocExtension_Internal_GetNumericProperty)(const char*, size_t, size_t*) = nullptr;
#endif

} // extern "C"

namespace folly::detail {

[[noreturn]] void ScopeGuardImplBase::terminate() noexcept {
    std::terminate();
}

} // namespace folly::detail
