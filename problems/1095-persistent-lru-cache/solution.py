from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):

        if key not in self.cache:
            return -1

        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, val):
        if key in self.cache:
            self.cache.move_to_end(key)
        
        self.cache[key] = val

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

def lru_cache_operations(capacity, ops):
    # ops is a list of operations such as ["put", k, v], ["get", k], ["crash"]
    # Return a list with the result of every "get" operation, in order.
    
    results = []
    active_cache = LRUCache(capacity)

    operation_log = []
    
    for op in ops:

        cmd = op[0]

        if cmd == 'put':
            key, val = op[1], op[2]
            active_cache.put(key, val)
            operation_log.append(["put", key, val])
        elif cmd == 'get':
            key = op[1]
            res = active_cache.get(key)
            results.append(res)
            operation_log.append(["get", key])
        elif cmd == "crash":
            active_cache = LRUCache(capacity)
            for log_op in operation_log:
                if log_op[0] == "put":
                    active_cache.put(log_op[1], log_op[2])
                elif log_op[0] == "get":
                    active_cache.get(log_op[1])
    
    return results


