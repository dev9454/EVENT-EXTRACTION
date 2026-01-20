import osmium
import shapely.wkb as wkblib
import geopandas as gpd
from time import time
from glob import glob
import os
import pickle
import shapely


class OsmHandler(osmium.SimpleHandler):
    def __init__(self, osm_file: str) -> None:
        osmium.SimpleHandler.__init__(self)

        self.osm_file = osm_file
        self.wkbfab = osmium.geom.WKBFactory()  # A global factory that creates WKB from a osmium geometry

        # dictionary with extracted scenarios, will be fulfilled at later stage
        self.output = {
            'bridges': [],
            'railway_crossing': [],
            'road_type': [],
            'barrier': [],
            'toll_booth': [],
            'power_line': [],
            'traffic_calming': [],
            'crossing': [],
            'traffic_signals': [],
            'roundabout': [],
            'tunnel': [],
            'traffic_sign': [],
            'gantry': [],
            'parking': [],
            'bus_stop': [],
            'manhole': [],
            'buildings': [],
            'pole': [],
            'street_lamp': [],
            'administrative_area': [],
            'landuse': [],
            'route_relations': []
        }

        # dictionary with additional tags to be extracted for each scenario, name of keys should match with self.output keys
        self.additional_tags = {
            'bridges': ['bridge', 'highway', 'level',
                        'lanes', 'layer', 'maxspeed', 'name', 'name:en', 'oneway',
                        'surface', 'bicycle', 'foot', 'covered',
                        'hov', 'hov:lanes', 'hov:lanes:backward',
                        'hov:lanes:forward', 'hov:conditional', 'hov:minimum', 'bus:lanes',
                        'access:lanes', 'motorcycle:lanes',
                        'turn:lanes', 'access'],
            'railway_crossing': ['railway'],
            'road_type': ['covered', 'highway', 'incline', 'lanes', 'layer',
                          'level', 'maxspeed', 'name', 'name:en', 'oneway',
                          'tunnel', 'surface', 'bicycle', 'foot',
                          'crossing', 'bridge', 'hov', 'hov:lanes', 'hov:lanes:backward',
                          'hov:lanes:forward', 'hov:conditional', 'hov:minimum', 'bus:lanes',
                          'access:lanes', 'motorcycle:lanes', 'turn:lanes',
                          'traffic_sign', 'busway', 'construction', 'access'],
            'barrier': ['barrier'],
            'toll_booth': ['barrier', 'name'],
            'power_line': ['cables', 'circuits', 'frequency',
                           'name', 'operator', 'power', 'voltage',
                           'layer', 'line'],
            'traffic_calming': ['traffic_calming',
                                'direction', 'surface'],
            'crossing': ['highway', 'crossing', 'crossing:island',
                         'crossing:markings', 'crossing:signals'],
            'traffic_signals': ['highway', 'traffic_signals',
                                'traffic_signals:direction', 'crossing', ],
            'roundabout': ['highway', 'junction', 'access', 'lanes'],
            'tunnel': ['covered', 'highway', 'incline', 'lanes', 'layer',
                       'level', 'maxspeed', 'name', 'name:en', 'oneway',
                       'tunnel', 'surface', 'bicycle', 'foot',
                       'hov', 'hov:lanes', 'hov:lanes:backward',
                       'hov:lanes:forward', 'hov:conditional', 'hov:minimum', 'bus:lanes',
                       'access:lanes', 'motorcycle:lanes', 'turn:lanes',
                       'traffic_sign', 'access'],
            'traffic_sign': ['highway', 'traffic_sign', 'name', 'maxspeed',
                             'direction', 'stop', 'traffic_sign:direction'],
            'gantry': ['man_made', 'gantry', 'gantry:type', 'traffic_signals',
                       'destination_sign', 'highway'],
            'parking': ['parking', 'surface', 'street_side', 'multi-storey',
                        'underground', 'lane', 'layby', 'on_kerb',
                        'rooftop', 'name', 'name:en', 'capacity', 'covered', 'access'],
            'bus_stop': ['highway', 'bus_stop', 'name', 'name:en'],
            'manhole': ['manhole'],
            'buildings': ['amenity', 'building', 'name',
                          'name:en', 'building:levels', 'alt_name', 'school'],
            'pole': ['power', 'man_made', 'material'],
            'street_lamp': ['highway', 'utility', 'support', 'lamp_mount', 'lamp_type'],
            'administrative_area': ['ISO3166-1', 'ISO3166-2', 'addr:city', 'addr:city:en',
                                    'addr:place', 'addr:place:en', 'admin_level', 'alt_name',
                                    'amenity', 'border_type', 'boundary', 'capital', 'governorate',
                                    'int_name', 'is_in:city', 'is_in:country', 'landuse',
                                    'loc_name', 'name', 'name:en', 'place', 'population',
                                    'postal_code', 'residential', 'state_code', 'timezone'
                                    ],
            'landuse': ['landuse', 'name', 'name:en', 'grass', 'residential', 'farmland',
                        'industrial', 'meadow', 'commercial', 'forest', 'retail', 'military',
                        'education', 'recreation', 'leisure', 'access', 'amenity', 'addr:city:en',
                        'shop', 'office', 'building', 'tourism', 'sport', 'surface'],
            'route_relations': ['name', 'type', 'network', 'ref', 'route', 'description',
                                'direction' 'toll', 'from', 'to', 'name:en']
        }

    def unpack_object(self, tag: object, node_name: str) -> dict:
        """
        Function to unpack osm objects to get geometry and tags that matches with self.additional_tags
        :param tag: osm object. Supported types: Area, Node, Way
        :type tag: object
        :param node_name: Name of scenario. Should match with self.additional_tags key names
        :type node_name: str
        :return: Unpacked osm object
        :rtype: dict
        """
        if isinstance(tag, osmium.osm.types.Area):  # unpack area
            obj_type = 'area'
            id = int(tag.orig_id())
            try:
                wkb = self.wkbfab.create_multipolygon(tag)
                geo = wkblib.loads(wkb, hex=True)
            except Exception as e:
                print(e)
                print(tag.id, {key: value for key, value in tag.tags})
                return {}
        elif isinstance(tag, osmium.osm.types.Node):  # unpack node
            obj_type = 'node'
            id = int(tag.id)
            try:
                wkb = self.wkbfab.create_point(tag)
                geo = wkblib.loads(wkb, hex=True)
            except Exception as e:
                print(e)
                print(tag.id, {key: value for key, value in tag.tags})
                return {}
        elif isinstance(tag, osmium.osm.types.Way):  # unpack way
            obj_type = 'way'
            id = int(tag.id)
            try:
                wkb = self.wkbfab.create_linestring(tag)
                geo = wkblib.loads(wkb, hex=True)
            except Exception as e:
                print(e)
                print(tag.id, {key: value for key, value in tag.tags})
                return {}
        else:
            print("OSM type not supported: ", type(tag))
            return {}
        # add OSM object id, its geometry, type and name
        row = {
            'el_id': id,
            'geometry': geo,
            'osm_type': obj_type,
            'object_name': node_name,
            'object_timestamp': tag.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        # update dict with all tags to have consistent output
        row.update({tag: '' for tag in self.additional_tags[node_name]})
        # update dirct with tags values for OSM objects
        row.update({key: value for key, value in tag.tags if key in self.additional_tags[node_name]})
        return row

    def get_bridges(self, way: object) -> None:
        """
        Function to extract bridges from OSM. https://wiki.openstreetmap.org/wiki/Key:bridge
        :param way: osm object
        :type way: object
        """
        bridges_tags = {
            'bridge': [
                'yes',
                'aqueduct',
                'cantilever',
                'covered',
                'low_water_crossing',
                'movable',
                'trestle',
                'viaduct'
            ],
            'building': ['bridge'],
            'man_made': ['bridge', 'goods_conveyor']
        }

        for key, value in bridges_tags.items():
            tag = way.tags.get(key)
            if tag in value:
                item = self.unpack_object(way, 'bridges')
                if item:
                    self.output['bridges'].append(item)

    def get_tunnel(self, way: object) -> None:
        """
        Function to extract tunnels from OSM
        :param way: osm object
        :type way: object
        """
        road_type_values = [
            'motorway',
            'trunk',
            'primary',
            'secondary',
            'tertiary',
            'unclassified',
            'residential',
            'living_street',
            'service',
            'road',
            'track',
            'motorway_link',
            'trunk_link',
            'primary_link',
            'secondary_link',
            'tertiary_link'
        ]

        if (('highway' in way.tags) and (way.tags['highway'] in road_type_values)) \
                and (
                (('tunnel' in way.tags) and (way.tags['tunnel'] in ('yes', 'building_passage',
                                                                    'covered', 'avalanche_protector'))) or
                (('covered' in way.tags) and (way.tags['covered'] == 'yes'))
        ):
            item = self.unpack_object(way, 'tunnel')
            if item:
                self.output['tunnel'].append(item)

    def get_road_type(self, way: object) -> None:
        """
        Function to extract road types  from OSM
        :param way: osm object
        :type way: object
        """
        road_type_values = [
            'motorway',
            'trunk',
            'primary',
            'secondary',
            'tertiary',
            'residential',
            'living_street',
            'motorway_link',
            'trunk_link',
            'primary_link',
            'secondary_link',
            'tertiary_link',
            'unclassified',
            'service',
            'road',
            'track'
        ]

        if ('highway' in way.tags) and (way.tags['highway'] in road_type_values):
            item = self.unpack_object(way, 'road_type')
            if item:
                self.output['road_type'].append(item)

    def get_barrier(self, way: object) -> None:
        """
        Function to extract barriers  from OSM. https://wiki.openstreetmap.org/wiki/Key:barrier
        :param way: osm object
        :type way: object
        """
        barrier_values = [
            'cable_barrier',
            'city_wall',
            'fence',
            'guard_rail',
            'hedge',
            'kerb',
            'wall',
            'yes',
            'jersey_barrier',
            'delineators',
            'chain',
            'bollard'
        ]

        if ('barrier' in way.tags) and (way.tags['barrier'] in barrier_values):
            item = self.unpack_object(way, 'barrier')
            if item:
                self.output['barrier'].append(item)

    def get_power_line(self, way: object) -> None:
        """
        Function to extract power lines  from OSM
        :param way: osm object
        :type way: object
        """
        if ('power' in way.tags) and (way.tags['power'] in ('line', 'minor_line')):
            item = self.unpack_object(way, 'power_line')
            if item:
                self.output['power_line'].append(item)

    def get_roundabout(self, way: object) -> None:
        """
        Function to extract roundabouts  from OSM
        :param way: osm object
        :type way: object
        """
        if (('junction' in way.tags) and (way.tags['junction'] == 'roundabout')) or \
                (('highway' in way.tags) and (way.tags['highway'] == 'mini_roundabout')):

            item = self.unpack_object(way, 'roundabout')
            if item:
                self.output['roundabout'].append(item)

    def get_railway_crossing(self, node: object) -> None:
        """
        Function to extract railway crossing with road at same level from OSM
        :param node: osm object
        :type node: object
        """

        if ('railway' in node.tags) and (node.tags['railway'] in ('level_crossing', 'tram_level_crossing')):
            item = self.unpack_object(node, 'railway_crossing')
            if item:
                self.output['railway_crossing'].append(item)

    def get_toll_booth(self, node: object) -> None:
        """
        Function to extract toll booth from OSM
        :param node: osm object
        :type node: object
        """

        if ('barrier' in node.tags) and (node.tags['barrier'] in ('toll_booth', 'border_control')):
            item = self.unpack_object(node, 'toll_booth')
            if item is not None:
                self.output['toll_booth'].append(item)

    def get_traffic_calming(self, node: object) -> None:
        """
        Function to extract road bumpers from OSM
        :param node: osm object
        :type node: object
        """
        tags = [
            'table',
            'hump',
            'bump',
            'cushion'
        ]

        if ('traffic_calming' in node.tags) and (node.tags['traffic_calming'] in tags):
            item = self.unpack_object(node, 'traffic_calming')
            if item:
                self.output['traffic_calming'].append(item)

    def get_crossing(self, node: object) -> None:
        """
        Function to extract pedestrian corssing from OSM
        :param node: osm object
        :type node: object
        """
        if ('highway' in node.tags) and (node.tags['highway'] == 'crossing'):
            item = self.unpack_object(node, 'crossing')
            if item is not None:
                self.output['crossing'].append(item)

    def get_traffic_lights(self, node: object) -> None:
        """
        Function to extract traffic lights from OSM
        :param node: osm object
        :type node: object
        """

        if ('highway' in node.tags) and (node.tags['highway'] == 'traffic_signals'):
            item = self.unpack_object(node, 'traffic_signals')
            if item is not None:
                self.output['traffic_signals'].append(item)

    def get_traffic_signs(self, node: object) -> None:
        """
        Function to extract traffic signs from OSM. https://wiki.openstreetmap.org/wiki/Key:traffic_sign
        :param node: osm object
        :type node: object
        """
        if (('highway' in node.tags) and (node.tags['highway'] in ('give_way', 'stop'))) or (
                'traffic_sign' in node.tags):
            item = self.unpack_object(node, 'traffic_sign')
            if item is not None:
                self.output['traffic_sign'].append(item)

    def get_gantry(self, node: object) -> None:
        """
        Function to extract gantry from OSM. https://wiki.openstreetmap.org/wiki/Key:traffic_sign
        :param node: osm object
        :type node: object
        """
        if (('man_made' in node.tags) and (node.tags['man_made'] == 'gantry')) or \
                ('gantry' in node.tags) or \
                (('highway' in node.tags) and (node.tags['highway'] == 'toll_gantry')):

            item = self.unpack_object(node, 'gantry')
            if item is not None:
                self.output['gantry'].append(item)

    def get_bus_stop(self, node: object) -> None:
        """
        Function to extract bus stops from OSM. https://taginfo.geofabrik.de/north-america/tags/highway=bus_stop#overview
        :param node: osm object
        :type node: object
        """

        if ('highway' in node.tags) and (node.tags['highway'] == 'bus_stop'):
            item = self.unpack_object(node, 'bus_stop')
            if item is not None:
                self.output['bus_stop'].append(item)

    def get_manhole(self, node: object) -> None:
        """
        Function to extract manhole from OSM. https://taginfo.geofabrik.de/europe/keys/manhole#overview
        :param node: osm object
        :type node: object
        """

        if 'manhole' in node.tags or \
                (('man_made' in node.tags) and (node.tags['man_made'] in ('reservoir_covered', 'manhole'))):
            item = self.unpack_object(node, 'manhole')
            if item is not None:
                self.output['manhole'].append(item)

    def get_pole(self, node: object) -> None:
        """
        Function to extract pole from OSM
        :param node: osm object
        :type node: object
        """

        if ('power' in node.tags and node.tags['power'] in ('tower', 'pole')) or \
                (('man_made' in node.tags) and (node.tags['man_made'] == 'utility_pole')):
            item = self.unpack_object(node, 'pole')
            if item is not None:
                self.output['pole'].append(item)

    def get_street_lamp(self, node: object) -> None:
        """
        Function to extract street lamps from OSM
        :param node: osm object
        :type node: object
        """

        if (('highway' in node.tags) and (node.tags['highway'] == 'street_lamp')) or \
                (('utility' in node.tags) and (node.tags['utility'] in ('street_lighting', 'power;street_lighting'))):
            item = self.unpack_object(node, 'street_lamp')
            if item is not None:
                self.output['street_lamp'].append(item)

    def get_parking(self, way: object) -> None:
        """
        Function to extract parkings from OSM. https://taginfo.geofabrik.de/europe/keys/parking#overview
        :param way: osm object
        :type way: object
        """

        if ('parking' in way.tags) or (('amenity' in way.tags) and (way.tags['amenity'] == 'parking')):
            item = self.unpack_object(way, 'parking')
            if item is not None:
                self.output['parking'].append(item)

    def get_buildings(self, way: object) -> None:
        """
        Function to extract schools, hospitals, police stations, fire stations station from OSM. https://taginfo.geofabrik.de/europe/keys/parking#overview
        :param way: osm object
        :type way: object
        """
        tags = ('school', 'hospital', 'fire_station', 'kindergarten', 'police', 'university', 'college', 'fuel', 'charging_station')
        if (('building' in way.tags) and (way.tags['building'] in tags)) or \
                (('amenity' in way.tags) and (
                        way.tags['amenity'] in tags)):
            item = self.unpack_object(way, 'buildings')
            if item:
                self.output['buildings'].append(item)

    def get_administrative_area(self, area: object) -> None:
        """
        Function to extract administrative areas / boundary  from OSM. https://taginfo.geofabrik.de/europe/keys/parking#overview
        :param area: osm object
        :type area: object
        """
        if ('boundary' in area.tags) and (area.tags['boundary'] == 'administrative') and ('admin_level' in area.tags):
            item = self.unpack_object(area, 'administrative_area')
            if item:
                self.output['administrative_area'].append(item)

    def get_landuse_area(self, area: object) -> None:
        """
        Function to extract administrative areas / boundary  from OSM. https://taginfo.geofabrik.de/europe/keys/parking#overview
        :param area: osm object
        :type area: object
        """
        if 'landuse' in area.tags:
            item = self.unpack_object(area, 'landuse')
            if item:
                self.output['landuse'].append(item)

    def node(self, n: object) -> None:
        """
        Function to process nodes
        :param n: OSM node
        :type n: object
        """
        if n.tags.__len__() > 0:
            self.get_traffic_calming(n)
            self.get_railway_crossing(n)
            self.get_toll_booth(n)
            self.get_crossing(n)
            self.get_traffic_lights(n)
            self.get_bus_stop(n)
            self.get_manhole(n)
            self.get_pole(n)
            self.get_street_lamp(n)

            # objects that could be ways and nodes
            self.get_barrier(n)
            self.get_gantry(n)
            self.get_traffic_signs(n)

    def way(self, w: object) -> None:
        """
        Function to process ways
        :param w: OSM way
        :type w: object
        """
        if w.tags.__len__() > 0:
            self.get_bridges(w)
            self.get_road_type(w)
            self.get_barrier(w)
            self.get_power_line(w)
            self.get_roundabout(w)
            self.get_tunnel(w)
            self.get_parking(w)

            # objects that could be ways and nodes
            self.get_traffic_calming(w)
            self.get_gantry(w)
            self.get_toll_booth(w)
            self.get_buildings(w)
            self.get_traffic_signs(w)

    def area(self, a: object) -> None:
        """
        Function to process areas
        :param a: OSM area
        :type a: object
        """
        if a.tags.__len__() > 0:
            self.get_administrative_area(a)
            self.get_landuse_area(a)
            if not a.from_way():  # extract additional objects if not from way (extracted already in way functions)
                self.get_parking(a)
                self.get_buildings(a)

    def relation(self, r: object) -> None:
        """
        Function to process relations
        :param r: OSM area
        :type r: object
        """

        if r.tags.__len__() > 0 and \
            (('type' in r.tags) and (r.tags['type'] == 'route')) and \
            (('route' in r.tags) and (r.tags['route'] == 'road')):
            row = {
                'el_id': int(r.id),
                'geometry': shapely.Point([0,0]),
                'type': 'relation',
                'object_name': 'route_relations',
                'object_timestamp': r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'members': ['{}|{}|{}'.format(m.type, m.ref, m.role) for m in r.members]
            }
            # update dict with all tags to have consistent output
            row.update({tag: '' for tag in self.additional_tags['route_relations']})
            # update dirct with tags values for OSM objects
            row.update({key: value for key, value in r.tags if key in self.additional_tags['route_relations']})
            self.output['route_relations'].append(row)

    def make_and_save_gdf(self, plot: bool = False, out_type: str = 'pkl') -> None:
        """
        Function to create GeoDataFrame from extracted osm objects in self.output and optionally save to html file as map
        :param plot: Create plot. Default False
        :type plot: bool
        :param out_type: Extension of output file. Supports gpkg and pkl. Default pkl
        :type out_type: str
        """
        for key, value in self.output.items():
            print(key)
            if not value or value == [{}]:
                print('Empty')
                continue
            gdf = gpd.GeoDataFrame(
                value, geometry='geometry', crs=4326
            )

            print(gdf.shape)
            if key == 'route_relations':
                gdf = gdf.explode(column='members')
                gdf[['member_type', 'member_id', 'member_role']] = gdf['members'].str.split('|', expand=True)
                gdf.drop(columns='members', inplace=True)
            output_name = self.osm_file.replace('.osm.pbf', "_{}".format(key))
            if plot:
                map = gdf.explore()
                map.save(output_name + '.html')

            if out_type in ('pkl', 'both'):
                with open(output_name + '.pkl', 'wb') as file:
                    pickle.dump(gdf, file)
            if out_type in ('gpkg', 'both'):
                gdf.to_file(output_name + '.gpkg', driver="GPKG")


if __name__ == '__main__':
    # path to file to local drive
    osm_file_location = r"C:\Users\qj0k6k\Documents\Thunder\OpenStreetMap\CES2025"
    osm_files = glob(os.path.join(osm_file_location,
                                  '*.osm.pbf'), recursive=True)

    for osm_file in osm_files:
        print(os.path.basename(osm_file))
        handler = OsmHandler(osm_file)
        # start data file processing
        print("Processing OSM file ...")
        start_time = time()
        handler.apply_file(osm_file, locations=True, idx='flex_mem')
        print("Processing OSM file took {:.2f} minutes".format((time() - start_time) / 60))
        print("Saving data to GPGK ...")
        handler.make_and_save_gdf(plot=True)
