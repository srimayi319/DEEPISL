Purpose: Document your goals, the problems you're solving, and your progress.

Content:

Objective: "To refactor the ISL recognition pipeline to be >95% accurate, eliminate prediction flickering, and build a robust foundation for new features."

Key Problems to Solve:

No spatial normalization (position/scale dependent).

Flawed temporal handling (padding + pooling).

No sign segmentation (constant noisy predictions).

Missing _IDLE_ class (causes false positives).

Low data volume (causes overfitting).
