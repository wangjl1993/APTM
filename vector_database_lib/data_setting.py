from weaviate.classes.config import Property, DataType



# 数据库每条数据属性，后续可根据需求增删
PROPERTIES = [
    Property(name="img_filename", data_type=DataType.TEXT),
    Property(name="parent", data_type=DataType.TEXT),        # 被去重后的图像所关联的parent
    Property(name="time", data_type=DataType.DATE),          # 拍摄时间, "1937-01-01T12:00:27.87+00:20"
    Property(name="location", data_type=DataType.TEXT)       # 拍摄地区
]

VECTOR = "vector"
UUID = "uuid"


