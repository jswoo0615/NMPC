#include <iostream>
#include <vector>
#include <cmath>

struct Waypoint {
    double x;
    double y;
};

struct VehicleState {
    double x;
    double y;
};

double distance(double x1, double y1, double x2, double y2) {
    return std::sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
}

// 차량 위치를 Frenet 좌표계로 변환
std::pair<double, double> toFrenet(const VehicleState& vehicle,
                                   const std::vector<Waypoint>& road) {
    double minDist = 1e9;
    double bestS = 0.0;
    double bestD = 0.0;

    // 누적 거리 계산용
    std::vector<double> cumulativeDist(road.size(), 0.0);
    for (size_t i = 1; i < road.size(); i++) {
        cumulativeDist[i] = cumulativeDist[i-1] +
            distance(road[i-1].x, road[i-1].y, road[i].x, road[i].y);
    }

    // 모든 segment에 대해 투영점 계산
    for (size_t i = 0; i < road.size()-1; i++) {
        double dx = road[i+1].x - road[i].x;
        double dy = road[i+1].y - road[i].y;
        double segLen = std::sqrt(dx*dx + dy*dy);

        double tangentX = dx / segLen;
        double tangentY = dy / segLen;

        // 차량 위치 벡터
        double vx = vehicle.x - road[i].x;
        double vy = vehicle.y - road[i].y;

        // 투영 길이 (도로 방향)
        double proj = vx * tangentX + vy * tangentY;

        // 투영점이 segment 범위 안에 있는지 확인
        if (proj >= 0 && proj <= segLen) {
            // 투영점 좌표
            double projX = road[i].x + proj * tangentX;
            double projY = road[i].y + proj * tangentY;

            // 차량과 투영점 사이 거리
            double distToProj = distance(vehicle.x, vehicle.y, projX, projY);

            if (distToProj < minDist) {
                minDist = distToProj;

                // s = 누적 거리 + proj
                bestS = cumulativeDist[i] + proj;

                // d = 법선 방향 거리
                double normalX = -tangentY;
                double normalY = tangentX;
                bestD = (vx * normalX + vy * normalY);
            }
        }
    }

    return {bestS, bestD};
}

int main() {
    // 예시 도로 중심선
    std::vector<Waypoint> road = {
        {0.0, 0.0}, {10.0, 2.0}, {20.0, 8.0}, {30.0, 18.0}
    };

    // 차량 위치
    VehicleState car = {15.0, 6.0};

    // Frenet 좌표 변환
    auto frenet = toFrenet(car, road);

    std::cout << "Frenet coordinates: s = " << frenet.first
              << ", d = " << frenet.second << std::endl;

    return 0;
}
