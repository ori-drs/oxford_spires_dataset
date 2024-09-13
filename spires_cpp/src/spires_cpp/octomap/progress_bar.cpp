#include "progress_bar.h"

void displayProgressBar(size_t current, size_t total) {
  const int barWidth = 50; // Width of the progress bar
  float progress = static_cast<float>(current) / total;
  int pos = static_cast<int>(barWidth * progress);

  std::cout << "[";
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << "%\r";
  std::cout.flush();
}
