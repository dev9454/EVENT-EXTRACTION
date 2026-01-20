import geopy
import argparse
from geopy.distance import distance
from geopy.point import Point
import math

def calculate_target_gps_position(host_lat_gps, host_long_gps, heading_angle, target_lat_pos, target_long_pos):
    # Host and Target position as Point object
    host_position = geopy.Point(host_lat_gps, host_long_gps)

    # Calculate the true bearing
    bearing = (heading_angle + math.degrees(math.tan(target_lat_pos / target_long_pos)) + 90) % 360

    # Calculate the distance between host and target
    distance_to_target = math.sqrt(target_long_pos ** 2 + target_lat_pos ** 2)

    # Calculate the new target position using the bearing and distance
    target_position = distance(meters=distance_to_target).destination(host_position, bearing)

    return target_position.latitude, target_position.longitude, bearing


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='calculate_target_gps_position')
    ap.add_argument("host_lat_gps", type=float, help="Enter the value of the variable that defines north/south")
    ap.add_argument("host_long_gps", type=float, help="Enter the value of the variable that defines east/west")
    ap.add_argument("heading_angle", type=float, help="Enter the value of the variable in which the host moves")
    ap.add_argument("target_lat_pos", type=float, help="Enter the value of the variable in which the target moves, which defines north/south")
    ap.add_argument("target_long_pos", type=float, help="Enter the value of the variable in which the target moves, which defines east/west")


    kwargs = vars(ap.parse_args())
    out = calculate_target_gps_position(kwargs['host_lat_gps'], kwargs['host_long_gps'], kwargs['heading_angle'], kwargs['target_lat_pos'], kwargs['target_long_pos'])
    print(out)

    