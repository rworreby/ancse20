#pragma once

double sign(double a) { return std::copysign(1.0, a); }

double minmod(double a, double b) {
    return 0.5 * (sign(a) + sign(b)) * std::min(std::abs(a), std::abs(b));
}

double slope_limiter(double sL, double sR) {
    return minmod(sL, sR);
}