#include "defns.i"
#include "extern.i"

// Replacement for unix's "round to nearest integer"

int rint (double x) {
  int i = (int) x;
  if (x >= 0.0) {
	int retval = ((x-i) >= 0.5) ? (i + 1) : (i);
    return retval;
  } else {
	int retval = (-x+i >= 0.5) ? (i - 1) : (i);
    return retval;
  }
}