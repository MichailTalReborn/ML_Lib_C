#include "arena_allocator/arena_main.c"
#include "rand_num_generator/rand.c"
#include <stdio.h>

int main(void) {
  mem_arena *perm_arena = arena_create(GiB(1), MiB(1));

  arena_destroy(perm_arena);
  return 0;
}
