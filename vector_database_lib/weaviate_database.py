import numpy as np
import weaviate 
import weaviate.classes as wvc
import torch
from typing import Dict, List, Any, Union
from vector_database_lib.data_setting import PROPERTIES, VECTOR, UUID


def init_weaviate_db(db_name: str, host: str="localhost", port: int=8080, grpc_port: int=50051, **kwargs):
    db_name = db_name.capitalize()
    client = weaviate.connect_to_local(host, port, grpc_port, **kwargs)
    if db_name not in client.collections.list_all():
        client.collections.create(db_name, properties=PROPERTIES)
    
    database = client.collections.get(db_name)
    return database, client


def write_data_to_database(database: weaviate.collections.Collection, items: List[Dict]):
    """_summary_

    Args:
        database (weaviate.collections.Collection): _description_
        items (List[Dict]): such as {
            'img_filename': '/x/x/x/a.jpg',
            'time': '1937-01-01T12:00:27.87+00:20',
            'vector': np.ndarray,
            ...
        }   more keys see {PROPERTIES}

    """

    non_property_list = [VECTOR, UUID]
    with database.batch.dynamic() as batch:
        for item in items:
            vector = item.get(VECTOR, None)
            uuid = item.get(UUID, None)
            properties = {k:v for k,v in item.items() if k not in non_property_list}
            batch.add_object(
                properties=properties,
                vector=vector,
                uuid=uuid
            )
        

def filter_func_generator(property_name: str, filter_name: str, params: Any):
    """ Generate weaviate filter func, such as 
            property_name='img_filename',
            filter_name='like',
            params='*0001*'
        then return: wvc.query.Filter.by_property("img_filename").like('*0001*')

    Args:
        property_name (_type_): _description_
        filter_name (_type_): _description_
        params (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert filter_name in [
            'contains_all', 'contains_any', 'equal', 'not_equal', 'greater_or_equal', 'greater_than',
            'less_or_equal', 'less_than', 'like', 'is_none', 'within_geo_range'
        ]
    
    return getattr(wvc.query.Filter.by_property(property_name), filter_name)(params)


def search_from_weaviate_db(
        database: weaviate.collections.Collection, vector: Union[List, np.ndarray, torch.Tensor], 
        top_k: int=None, filters: Any=None, **kwargs
    ) -> List[Dict]:
    """Retrieval person information from weaviate database according by near embedding vector.

    Args:
        database (weaviate.collections.Collection): _description_
        vector (Union[List, np.ndarray, torch.Tensor]): _description_
        top_k (int): _description_
        filters (Any, optional): _description_. Defaults to None.

    Returns:
        List[Dict]: _description_
    """
    if isinstance(vector, torch.Tensor):
        vector = vector.cpu().numpy().tolist()
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    
    response = database.query.near_vector(
        near_vector=vector,
        limit=top_k,
        return_metadata=wvc.query.MetadataQuery(distance=True),
        filters=filters,
        **kwargs
    )

    results = []
    for obj in response.objects:
        score = 1 - obj.metadata.distance
        results.append({
            **obj.properties, 'score':score
        })
        

    return results




# weaviate 3.+ version
"""
def build_weaviate_client(db_name, url='http://localhost:8080'):
    client = weaviate.Client(url=url)
    class_obj = {
        'class': db_name,
        'vectorIndexConfig':{
        'distance': 'cosine',   # 这里的distance是选择向量检索方式，这里选择的是cosine距离
        },
    }
    client.schema.create_class(class_obj)
    return client


def write_data_to_database(client: weaviate.Client, db_name: str, feats: np.ndarray, properties: Dict, batch_size: int=100):
    n = len(feats)
    properties_key = properties.keys()
    properties_value = properties.values()

    with client.batch(batch_size=batch_size) as batch:
        for i in range(n):
            one_vector = feats[i]
            one_vector_properties = [value[i] for value in properties_value]
            batch.add_data_object(
                data_object=dict(zip(properties_key, one_vector_properties)),
                class_name=db_name,
                vector=one_vector
            )
"""