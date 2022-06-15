# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from locust import HttpUser, task, between, events
import time
import psutil
import atexit
import json
import datetime
import pytz
from influxdb import InfluxDBClient
import socket
hostname=socket.gethostname()

client = InfluxDBClient(host="127.0.0.1", port="8086")
client.switch_database('home')
# class MyTasks(task):


class QuickstartUser(HttpUser):
    # tasks = [MyTasks]
    wait_time = between(1, 5)
    sock = None

    def exit_handler(self):
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()

    def hook_request_success(self, request_type, name, response_time, response_length):
        message = "%s %d %d\n" % ("performance." + name.replace('.', '-'), response_time, time.time())
        self.sock.send(message.encode())

    def hook_request_fail(self, request_type, name, response_time, exception):
        self.request_fail_stats.append([name, request_type, response_time, exception])
    
    def individual_success_handle(self, request_type, name, response_time, response_length, **kwargs):
        disk = psutil.disk_usage('/')
        mem = psutil.virtual_memory()
        load = psutil.getloadavg()
        SUCCESS_TEMPLATE = '[{"measurement": "%s","tags": {"hostname":"%s","requestName": "%s","requestType": "%s","status":"%s"' \
                       '},"time":"%s","fields": {"responseTime": "%s","responseLength":"%s","load_1": "%s","load_5": "%s","load_15": "%s","disk_percent":"%s","disk_free": "%s","disk_used": "%s","mem_percent": "%s","mem_free": "%s","mem_used": "%s"}' \
                       '}]'
        json_string = SUCCESS_TEMPLATE % (
        "ResponseTable", hostname, name, request_type, "success", datetime.datetime.now(tz=pytz.UTC), response_time,
        response_length,
        load[0],
        load[1],
        load[2],
        disk.percent,
        disk.free,
        disk.used,
        mem.percent,
        mem.free,
        mem.used)
        client.write_points(json.loads(json_string), time_precision='ms')
        #print(json_string)



    def individual_fail_handle(self, request_type, name, response_time, response_length, exception, **kwargs):
        disk = psutil.disk_usage('/')
        mem = psutil.virtual_memory()
        load = psutil.getloadavg()
        FAIL_TEMPLATE = '[{"measurement": "%s","tags": {"hostname":"%s","requestName": "%s","requestType": "%s","exception":"%s","status":"%s"' \
                    '},"time":"%s","fields": {"responseTime": "%s","responseLength":"%s","load_1": "%s","load_5": "%s","load_15": "%s","disk_percent":"%s","disk_free": "%s","disk_used": "%s","mem_percent": "%s","mem_free": "%s","mem_used": "%s"}' \
                    '}]'
        json_string = FAIL_TEMPLATE % (
        "ResponseTable", hostname, name, request_type, exception, "fail", datetime.datetime.now(tz=pytz.UTC),
        response_time, response_length,
        load[0],
        load[1],
        load[2],
        disk.percent,
        disk.free,
        disk.used,
        mem.percent,
        mem.free,
        mem.used)
        client.write_points(json.loads(json_string), time_precision='ms')
        #print(json_string)

    def __init__(self, parent):
        super().__init__(parent)
        self.sock = socket.socket()
        self.sock.connect(("localhost", 2003))
        events.request_success.add_listener(self.hook_request_success)
        events.request_failure.add_listener(self.hook_request_fail)
        
        events.request_success.add_listener(self.individual_success_handle)
        events.request_failure.add_listener(self.individual_fail_handle)

        atexit.register(self.exit_handler)

    @task(2)
    def index(self):
        self.client.get("/")
    @task(1)
    def index(self):
        self.client.get("/predict")
        
