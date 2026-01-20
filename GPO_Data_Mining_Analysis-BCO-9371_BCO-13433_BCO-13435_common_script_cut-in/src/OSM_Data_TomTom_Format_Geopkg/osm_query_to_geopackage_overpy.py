import geopandas
import overpy
import numpy as np
from shapely.geometry import LineString, Point
from fiona.crs import from_epsg
from shapely.wkt import loads


class LogBoundingBox:

    def __init__(self):
        self.BoundingBox = dict()
        self.BoundingBox["min_lat"] = 0
        self.BoundingBox["min_lon"] = 0
        self.BoundingBox["max_lat"] = 0
        self.BoundingBox["max_lon"] = 0
        

    def return_bbox(self):
        return self.BoundingBox 

   
    def generate_bbox(self):
        # crooks road and west south boulevard - bounding box location
        # lat = [42.62317,42.62179]
        # lon = [-83.16705,-83.17023]        
        # self.BoundingBox["west"] = min(lat)
        # self.BoundingBox["south"] = min(lon)
        # self.BoundingBox["east"] = max(lat)
        # self.BoundingBox["north"] =max(lon)

        # west south boulevard to west long lake road - bounding box location
        lat = [42.590749,42.621791,42.606270,42.604718]
        lon = [-83.168450,-83.169934,-83.149757,-83.188964]
        self.BoundingBox["west"] = min(lat)
        self.BoundingBox["south"] = min(lon)
        self.BoundingBox["east"] = max(lat)
        self.BoundingBox["north"] =max(lon)

class OpenStreetMapAPI:

    def __init__(self):       
        self.api_client = overpy.Overpass()    

    
    
    def get_map_by_bbox(self,bbox):  
        # generate map query for nodes, ways, and relations   
        tag_query =str(f' [out:json][timeout:300];(nwr({bbox["west"]},{bbox["south"]},{bbox["east"]},{bbox["north"]}); ); out geom;')  
        # overpy library used for osm data query    
        nwr_map_data = self.api_client.query(tag_query)
        return nwr_map_data
        
class OsmDataProcessing:

    def __init__(self, osm_data):
        self.osm_map = osm_data
        self.roadNodeOverpy = {'lat': np.array([]), 'lon': np.array([])}
        self.speedRestriction = dict() 
        self.roadNetworkArc = dict()

    def generate_nodeOverpy(self):
        lat_list = []
        lon_list = []

        #iterate through each node from querried data for lat and lon value
        for node in self.osm_map.nodes:
            lat_list.append(float(node.lat))
            lon_list.append(float(node.lon))
            

        self.roadNodeOverpy['lat'] = np.array(lat_list,dtype=float)
        self.roadNodeOverpy['lon'] = np.array(lon_list,dtype=float)                             



    def generate_speedRestrictionOnCenterLine(self):  
        speedRestrictionOnCenterLine_dict= dict()
        maxspeed_list=[]
        id_list=[]
        source_list=[]
        unit_list=[]
        lat_list=[]
        lon_list=[]
        verDate_list=[]
        node_found=False

        # iterate through each way from querried data for each way
        for way in self.osm_map.ways:   
            # filter the tags for speed information
            if 'maxspeed' in way.tags:              
                # extract lat lon position of speed information tag using node ids
                for way_node_id in way._node_ids:
                   
                    for node in self.osm_map.nodes:
                        # if one of the node id is matched from the list of all node ids related to speed information,
                        # # extract the lat and lon value
                        if way_node_id == node.id:                           
                            lat_list.append(node.lat)                           
                            lon_list.append(node.lon)
                            node_found=True
                            break

                    if node_found:
                        node_found=False
                        break
                 
                id_list.append(way.id)
                source_list.append("osm")
                maxspeed_list.append(way.tags['maxspeed'])
                unit_list.append("mph")
                verDate_list.append("11/2/2023")
      
        speedRestrictionOnCenterLine_dict["id"]=id_list
        speedRestrictionOnCenterLine_dict["source"]=source_list
        speedRestrictionOnCenterLine_dict["lat"]=np.array(lat_list,dtype=float)
        speedRestrictionOnCenterLine_dict["lon"]=np.array(lon_list,dtype=float)
        speedRestrictionOnCenterLine_dict["restrInfos"]=maxspeed_list
        speedRestrictionOnCenterLine_dict["verDate"]=verDate_list
        speedRestrictionOnCenterLine_dict["speedUnit"]=unit_list
                
        return speedRestrictionOnCenterLine_dict
        
    def return_roadNode(self):
        return self.roadNodeOverpy    
    
    def generate_centerLine(self):
        id_list=[]
        source_list=[]
        type_list=[]
        lat_list=[]
        lon_list=[]
        line_list=[]
        length_list=[]
        roadArea_list=[]
        headNode_list=[]
        tailNode_list=[]
        fingerPr_list=[]  

        # iterate through each way from querried data for each way
        for way in self.osm_map.ways:           
            lat_list = []
            lon_list = [] 
            total_nodes=0   
            #iterate through all the node ids within each way 
            for way_node_id in way._node_ids:                                 
                for node in self.osm_map.nodes:  
                    if way_node_id == node.id: 
                        total_nodes += 1                                                                
                        lat_list.append(node.lat)
                        lon_list.append(node.lon)   
            #create a list of all nodes for each way to create a linestring geometry
            node_array=[]
            for i in range(len(lat_list)):                                               
                p1 = Point(lon_list[i], lat_list[i])
                node_array.append(p1)

            #if each way has more than two nodes then create the line/roac arc using Linestring python library
            if len(node_array)>1 and total_nodes>1:          
                centerLine = LineString(node_array)
                line_list.append(centerLine)          
                id_list.append(way.id)
                type_list.append("ROAD_ARC")
                length_list.append(0)
                roadArea_list.append(0)
                headNode_list.append(0)
                tailNode_list.append(0)
                fingerPr_list.append(0)
                source_list.append("osm")              
                                
        self.roadNetworkArc["id"]=id_list
        self.roadNetworkArc["line"]=line_list   
        self.roadNetworkArc["type"]=type_list          #TODO : change the input source as per the type 
        self.roadNetworkArc["length"]=length_list      #TODO : change the input source as per the calculated len
        self.roadNetworkArc["roadArea"]=roadArea_list  #TODO : change the input source as per the gpkg table 
        self.roadNetworkArc["headNode"]=headNode_list  #TODO : change the input source as per the gpkg table 
        self.roadNetworkArc["tailNode"]=tailNode_list  #TODO : change the input source as per the gpkg table 
        self.roadNetworkArc["fingerPr"]=fingerPr_list  #TODO : change the input source as per the gpkg table 
        self.roadNetworkArc["source"]=source_list                
        
        return self.roadNetworkArc   


    def generate_singleBorder(self):
        singleBorder_dict= dict()  
        id_list=[]
        index_list=[]
        lat_list=[]
        lon_list=[]
        type_list=[]
        color_list=[]
        node_found=False

        # iterate through each way from querried data for each way
        for way in self.osm_map.ways:       
            # filter the tags for tags
            if 'barrier' in way.tags:                     
                # extract lat lon position of each barrier tag using node ids
                for way_node_id in way._node_ids:
                    # extract lat lon position of speed information tag using node ids
                    for node in self.osm_map.nodes:
                        # if one of the node id is matched from the list of all node ids related to barrier information,
                        # extract the lat and lon value
                        if way_node_id == node.id:                           
                            lat_list.append(node.lat)                           
                            lon_list.append(node.lon)                            
                            node_found=True
                            break

                    if node_found:
                        node_found=False
                        break
                    
                id_list.append(way.id)
                index_list.append("0") #TODO : in future change the input as per the number of border for lanegroup
                type_list.append(way.tags['barrier'])
                color_list.append("tbd")
        # iterate through each node from querried data for each way
        for node in self.osm_map.nodes:      
            if 'barrier' in node.tags: 
                lat_list.append(node.lat)                           
                lon_list.append(node.lon)                     
                id_list.append(node.id)
                index_list.append("1") #TODO : in future change the input as per the number of border for lanegroup
                type_list.append(node.tags['barrier'])
                color_list.append("tbd")

        #there is no barrier in relations 
      
        singleBorder_dict["id"]=id_list
        singleBorder_dict["lat"]=np.array(lat_list,dtype=float)
        singleBorder_dict["lon"]=np.array(lon_list,dtype=float)
        singleBorder_dict["index"]=index_list
        singleBorder_dict["type"]=type_list
        singleBorder_dict["color"]=color_list

                
        return singleBorder_dict


if __name__ == '__main__':    
    bbox_obj = LogBoundingBox()
    bbox_obj.generate_bbox()    
    bbox_dict = bbox_obj.return_bbox()    
    osm_api = OpenStreetMapAPI()
    osm_data = osm_api.get_map_by_bbox(bbox_dict)    
    osm_processor = OsmDataProcessing(osm_data)   


    osm_processor.generate_nodeOverpy()   

    # road network nodes
    roadNode = osm_processor.return_roadNode()   
    node_geodf = geopandas.GeoDataFrame(roadNode, geometry=geopandas.points_from_xy(roadNode["lon"], roadNode["lat"]))    
    node_geodf.to_file("OVERPY_roadNode.gpkg", driver='GPKG', layer="roadNode")

    #road network speed limits
    speedRestrictionOnCenterLine = osm_processor.generate_speedRestrictionOnCenterLine() 
    speedRestrictionOnCenterLine_geodf = geopandas.GeoDataFrame(speedRestrictionOnCenterLine,geometry=geopandas.points_from_xy(speedRestrictionOnCenterLine["lon"], speedRestrictionOnCenterLine["lat"]))    
    speedRestrictionOnCenterLine_geodf.to_file("OVERPY_speedRestrictionOnCenterLine.gpkg", driver='GPKG', layer="speedRestrictionOnCenterLine")

    #road netwrok lines
    roadNetworkArc_dict = osm_processor.generate_centerLine()     
    roadNetworkArc_geodf = geopandas.GeoDataFrame(roadNetworkArc_dict, geometry='line')    
    roadNetworkArc_geodf.to_file("OVERPY_roadNetworkArc.gpkg", driver='GPKG', layer="roadNetworkArc")

    # road network barriers
    singleBorder = osm_processor.generate_singleBorder()   
    singleBorder_geodf = geopandas.GeoDataFrame(singleBorder,geometry=geopandas.points_from_xy(singleBorder["lon"], singleBorder["lat"]))
    singleBorder_geodf.to_file("OVERPY_singleBorder.gpkg", driver='GPKG', layer="singleBorder")


    
    
    
    







