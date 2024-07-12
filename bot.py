# SPDX-License-Identifier: BSD-3-Clause
import random

# flake8: noqa F401
from collections.abc import Callable

import numpy as np

from vendeeglobe import Checkpoint, Heading, Instructions, Location, Vector, config
from vendeeglobe.utils import distance_on_surface, goto, wrap, wind_force

import math
import heapq


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        try:
            return heapq.heappop(self.elements)[1]
        except IndexError:
            return None


def calculate_new_position(lat, lon, heading, distance):
    """
    Calculate the new latitude and longitude after traveling a certain distance from a starting point.

    Parameters:
    lat (float): Latitude of the starting point in degrees
    lon (float): Longitude of the starting point in degrees
    heading (float): Heading in degrees (0 is east, 90 is north, 180 is west, 270 is south)
    distance (float): Distance to travel in kilometers

    Returns:
    tuple: New latitude and longitude in degrees
    """
    # Convert latitude, longitude, and adjusted heading to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Adjust the heading to match the new system (East = 0, North = 90, West = 180, South = 270)
    adjusted_heading = (90 - heading) % 360
    heading_rad = math.radians(adjusted_heading)

    # Radius of the Earth in kilometers
    R = 6371

    # Calculate the new latitude
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(distance / R)
        + math.cos(lat_rad) * math.sin(distance / R) * math.cos(heading_rad)
    )

    # Calculate the new longitude
    new_lon_rad = lon_rad + math.atan2(
        math.sin(heading_rad) * math.sin(distance / R) * math.cos(lat_rad),
        math.cos(distance / R) - math.sin(lat_rad) * math.sin(new_lat_rad),
    )

    # Convert the new latitude and longitude from radians to degrees
    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)

    return new_lat, new_lon


def ray(start, goal, point_distance=0.04394) -> tuple[np.array, np.array]:
    """
    Compute lats, lons arrays from start to goal that are separated by 0.04394 degrees
    """
    start_lat, start_lon = start
    goal_lat, goal_lon = goal

    dlat = goal_lat - start_lat
    dlon = goal_lon - start_lon

    # Calculate the number of points to generate
    n_points = int((abs(dlat) + abs(dlon)) / point_distance)

    # Calculate the latitude and longitude differences
    dlat_point = (dlat) / n_points
    dlon_point = (dlon) / n_points

    # Generate the lats and lons arrays
    lats = [start_lat + i * dlat_point for i in range(n_points)]
    lons = [start_lon + i * dlon_point for i in range(n_points)]

    return lats, lons


def get_neighbors(
    latitude,
    longitude,
    heading,
    world_map,
    step_distance=50.0,
    degrees=(0, 5, -5, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180),
):
    potential_locations = [
        wrap(
            *calculate_new_position(
                latitude, longitude, (heading + degree) % 360, step_distance
            )
        )
        for degree in degrees
    ]
    potential_locations = [
        (float(location[0]), float(location[1])) for location in potential_locations
    ]
    potential_locations = [
        location
        for location in potential_locations
        if (lambda lats, lons: world_map(latitudes=lats, longitudes=lons).all())(
            *ray((latitude, longitude), location)
        )
    ]
    return potential_locations


def are_we_there_yet(latitude1, longitude1, latitude2, longitude2, radius=50.0):
    dist_to_finish = distance_on_surface(
        longitude1=longitude1,
        latitude1=latitude1,
        longitude2=longitude2,
        latitude2=latitude2,
    )
    return dist_to_finish < radius


LOOKAHEAD_THRESHOLD = 5 * 24
DEFAULT_SPEED = 70.0


def heuristic(start, goal, forecast):
    # heuristic is estimated time to the finish with speed estimated at start
    dist = distance_on_surface(
        latitude1=start[0],
        longitude1=start[1],
        latitude2=goal[0],
        longitude2=goal[1],
    )

    ship_vector = vector_from_points(start, goal)
    wind = np.array(forecast(latitudes=start[0], longitudes=start[1], times=0))
    vec = wind_force(ship_vector, wind)
    # km / h
    speed = np.linalg.norm(vec)
    duration = dist / speed
    return duration


def vector_from_points(start, goal):
    heading = goto(Location(start[1], start[0]), Location(goal[1], goal[0]))
    h = heading * np.pi / 180.0
    return np.array([np.cos(h), np.sin(h)])


def cost(start, goal, time, forecast):
    if start == goal:
        return 0.0
    # km
    dist = distance_on_surface(
        latitude1=start[0],
        longitude1=start[1],
        latitude2=goal[0],
        longitude2=goal[1],
    )
    ship_vector = vector_from_points(start, goal)
    wind = np.array(forecast(latitudes=start[0], longitudes=start[1], times=time))
    vec = wind_force(ship_vector, wind)
    # km / h
    speed = np.linalg.norm(vec)
    duration = dist / speed
    return duration


def astar(latitude, longitude, checkpoint, world_map, forecast, start_heading):
    latitude = float(latitude)
    longitude = float(longitude)
    start = (latitude, longitude)
    frontier = PriorityQueue()
    frontier.put(start, 0.0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    finish = (checkpoint.latitude, checkpoint.longitude)
    loops = 0

    finish_found = False

    while (
        (current := frontier.get())
        and cost_so_far[current] < LOOKAHEAD_THRESHOLD
        and loops < 5000
    ):
        if are_we_there_yet(*current, *finish):
            finish_found = True
            break

        if came_from[current] is not None:
            new_heading = goto(
                Location(came_from[current][1], came_from[current][0]),
                Location(current[1], current[0]),
            )
        else:
            new_heading = start_heading
        dist_to_finish = distance_on_surface(
            latitude1=current[0],
            longitude1=current[1],
            latitude2=finish[0],
            longitude2=finish[1],
        )
        dist_from_start = distance_on_surface(
            latitude1=current[0],
            longitude1=current[1],
            latitude2=start[0],
            longitude2=start[1],
        )
        step_distance = min(1500.0, dist_to_finish, max(dist_from_start, 100))
        neighbors = get_neighbors(
            current[0], current[1], new_heading, world_map, step_distance=step_distance
        )
        for next_location in neighbors:
            new_cost = cost_so_far[current] + cost(
                current, next_location, cost_so_far[current], forecast
            )
            if (
                next_location not in cost_so_far
                or new_cost < cost_so_far[next_location]
            ):
                cost_so_far[next_location] = new_cost
                priority = new_cost + heuristic(next_location, finish, forecast)
                frontier.put(next_location, priority)
                came_from[next_location] = current
            loops += 1

    path = []
    if not finish_found:
        # find the closes point to the finish
        # closest, _ = min(
        #     heapq.nsmallest(50, cost_so_far.items(), key=lambda item: abs(finish[0] - item[0][0]) + abs(finish[1] - item[0][1])),
        #     key=lambda item: distance_on_surface(
        #         latitude1=item[0][0],
        #         longitude1=item[0][1],
        #         latitude2=finish[0],
        #         longitude2=finish[1],
        #     )
        #     / DEFAULT_SPEED
        #     + item[1],
        # )
        # current = closest

        # fall back to straight line
        path = [finish]
    else:
        while current and current != start:
            path.append(current)
            current = came_from[current]
    path.reverse()

    return path


class Bot:
    """
    This is the ship-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = "Artimi"  # This is your team name
        # This is the course that the ship has to follow
        self.course = [
            Checkpoint(latitude=18.3073, longitude=-68.2965, radius=50),
            Checkpoint(latitude=17.7821, longitude=-68.4338, radius=50),
            Checkpoint(latitude=9.421, longitude=-80.288, radius=50),
            Checkpoint(latitude=9.0632, longitude=-79.6976, radius=50),
            Checkpoint(latitude=8.6127, longitude=-79.2389, radius=50),
            Checkpoint(latitude=6.898, longitude=-79.942, radius=50),
            Checkpoint(latitude=6.843, longitude=-80.338, radius=50),
            Checkpoint(
                latitude=2.806318, longitude=-168.943864, radius=2000.0
            ),  # checkpoint 1
            # unders australia
            # Checkpoint(latitude=-14.28, longitude=177.45, radius=50),
            # Checkpoint(latitude=-44.461, longitude=146.755, radius=50.0),
            # Checkpoint(latitude=-35.145, longitude=114.478, radius=50.0),
            Checkpoint(latitude=4.561, longitude=125.903, radius=50.0),
            Checkpoint(latitude=0.937, longitude=119.465, radius=50.0),
            Checkpoint(latitude=-4.662, longitude=116.895, radius=50.0),
            Checkpoint(latitude=-5.689, longitude=106.161, radius=20.0),
            Checkpoint(latitude=-6.06, longitude=105.634, radius=20.0),
            Checkpoint(latitude=-6.377, longitude=105.403, radius=50.0),
            Checkpoint(
                latitude=-15.668984, longitude=77.674694, radius=1200.0
            ),  # checkpoint 2
            Checkpoint(latitude=12.2937, longitude=51.5753, radius=50.0),
            Checkpoint(latitude=12.585, longitude=43.358, radius=50.0),
            Checkpoint(latitude=12.8355, longitude=43.19, radius=50.0),
            Checkpoint(latitude=27.41, longitude=34.217, radius=50.0),
            Checkpoint(latitude=28.501, longitude=33.351, radius=10.0),
            Checkpoint(latitude=29.649, longitude=32.637, radius=10.0),
            Checkpoint(latitude=30.6178, longitude=32.4553, radius=10.0),
            Checkpoint(latitude=31.806, longitude=32.4, radius=10.0),
            Checkpoint(latitude=34.558, longitude=25.269, radius=50.0),
            Checkpoint(latitude=36.841, longitude=13.557, radius=50.0),
            Checkpoint(latitude=37.819, longitude=9.009, radius=50.0),
            Checkpoint(latitude=36.434, longitude=-1.89, radius=50.0),
            Checkpoint(latitude=35.959, longitude=-5.482, radius=50.0),
            Checkpoint(latitude=35.968, longitude=-6.213, radius=50.0),
            Checkpoint(latitude=36.993, longitude=-9.286, radius=10.0),
            Checkpoint(latitude=38.94, longitude=-9.679, radius=50.0),
            Checkpoint(latitude=44.0, longitude=-9.514, radius=20.0),
            Checkpoint(
                latitude=config.start.latitude,
                longitude=config.start.longitude,
                radius=5,
            ),
        ]
        self._last_location = None
        self._path = []
        self._path_age = 0.0

    def run(
        self,
        t: float,
        dt: float,
        longitude: float,
        latitude: float,
        heading: float,
        speed: float,
        vector: np.ndarray,
        forecast: Callable,
        world_map: Callable,
    ) -> Instructions:
        """
        This is the method that will be called at every time step to get the
        instructions for the ship.

        Parameters
        ----------
        t:
            The current time in hours.
        dt:
            The time step in hours.
        longitude:
            The current longitude of the ship.
        latitude:
            The current latitude of the ship.
        heading:
            The current heading of the ship.
        speed:
            The current speed of the ship.
        vector:
            The current heading of the ship, expressed as a vector.
        forecast:
            Method to query the weather forecast for the next 5 days.
            Example:
            current_position_forecast = forecast(
                latitudes=latitude, longitudes=longitude, times=0
            )
        world_map:
            Method to query map of the world: 1 for sea, 0 for land.
            Example:
            current_position_terrain = world_map(
                latitudes=latitude, longitudes=longitude
            )

        Returns
        -------
        instructions:
            A set of instructions for the ship. This can be:
            - a Location to go to
            - a Heading to point to
            - a Vector to follow
            - a number of degrees to turn Left
            - a number of degrees to turn Right

            Optionally, a sail value between 0 and 1 can be set.
        """
        # Initialize the instructions
        instructions = Instructions()

        if (
            speed <= 0.1
            or self._last_location is not None
            and self._last_location == (latitude, longitude)
        ):
            neighbors = get_neighbors(
                latitude,
                longitude,
                heading,
                world_map,
                degrees=tuple(range(0, 360, 30)),
            )
            next_location = random.choice(neighbors)

            new_heading = goto(
                Location(longitude, latitude),
                Location(next_location[1], next_location[0]),
            )
            instructions.heading = Heading(angle=new_heading)
            self._last_location = (latitude, longitude)
            self._path.clear()
            return instructions

        # Go through all checkpoints and find the next one to reach
        for ch in self.course:
            dist = distance_on_surface(
                longitude1=longitude,
                latitude1=latitude,
                longitude2=ch.longitude,
                latitude2=ch.latitude,
            )
            # Compute the distance to the checkpoint
            # Check if the checkpoint has been reached
            if dist < ch.radius:
                ch.reached = True
                self._path.clear()
            if not ch.reached:
                break

        probe_location = calculate_new_position(
            latitude, longitude, heading, min(150, dist)
        )
        probe_lats, probe_lons = ray((latitude, longitude), probe_location)
        if not world_map(latitudes=probe_lats, longitudes=probe_lons).all():
            self._path.clear()

        if (
            not self._path
            or are_we_there_yet(
                latitude, longitude, self._path[0][0], self._path[0][1], radius=250.0
            )
            or t - self._path_age > LOOKAHEAD_THRESHOLD
        ) and dist < 5000:
            self._path = astar(latitude, longitude, ch, world_map, forecast, heading)
            self._path_age = t

        try:
            next_location = self._path[0]
        except IndexError:
            # fallback to the straight line
            next_location = (ch.latitude, ch.longitude)
        new_heading = goto(
            Location(longitude, latitude),
            Location(next_location[1], next_location[0]),
        )
        instructions.heading = Heading(angle=new_heading)

        self._last_location = (latitude, longitude)
        return instructions
