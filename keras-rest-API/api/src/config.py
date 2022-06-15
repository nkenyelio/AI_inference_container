# initialize constants used for server queuing
MHEALTH_QUEUE = "mhealth_queue"
BATCH_SIZE = 32
WORKER_SLEEP = 0.25
CONSUMER_SLEEP = 0.25
# initialize Redis connection settings
REDIS_HOST = "redis"
REDIS_PORT = 6379
REDIS_DB = 0

# weights files
#WEIGHTS_JSON = "/api/trained_model/trained_model.json"
WEIGHTS_H5 = "/api/trained_model/lstm.h5"

# data to be analyzed

INPUT_RAW_DATAX = "/api/data/dataX.npy"
INPUT_RAW_DATAY = "/api/data/dataY.npy"
FILE_PATH = "/api/data/mHealth_subject.log"
# logging location
LOG_DIR = "/api/logs"