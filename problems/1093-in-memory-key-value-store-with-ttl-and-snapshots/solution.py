def kv_store(commands):
    # commands is a list of operations.
    # Process them in order and return the collected output as a single
    # string, with individual output lines joined by '\n'.
    current_time = 0
    store = {}       
    snapshots = {}    
    output = []

    def cleanup_expired():
        expired_keys = [
            k for k, v in store.items() 
            if v['expiry'] is not None and current_time >= v['expiry']
        ]
        for k in expired_keys:
            del store[k]

    for cmd in commands:
        op = cmd[0]
        
        if op == "TICK":
            current_time += int(cmd[1])
            
        elif op == "SET":
            key, value = cmd[1], cmd[2]
            if len(cmd) == 4:
                ttl = int(cmd[3])
                expiry = current_time + ttl
            else:
                expiry = None 
            store[key] = {'value': value, 'expiry': expiry}
            
        elif op == "GET":
            key = cmd[1]
            cleanup_expired()
            if key in store:
                output.append(str(store[key]['value']))
            else:
                output.append("NULL")
                
        elif op == "DELETE":
            key = cmd[1]
            cleanup_expired()
            if key in store:
                del store[key]
                output.append("true")
            else:
                output.append("false")
                
        elif op == "SCAN":
            prefix = cmd[1]
            cleanup_expired()
            matches = []
            for k, v in store.items():
                if k.startswith(prefix):
                    matches.append((k, v['value']))
            
            if matches:
                matches.sort(key=lambda x: x[0])
                matches_str = ", ".join(f"{k}={v}" for k, v in matches)
                output.append(matches_str)
            else:
                output.append("(empty)")
                
        elif op == "BACKUP":
            name = cmd[1]
            cleanup_expired()
            snap = {}
            for k, v in store.items():
                rem_ttl = None if v['expiry'] is None else v['expiry'] - current_time
                snap[k] = {'value': v['value'], 'remaining_ttl': rem_ttl}
            snapshots[name] = snap
            output.append(str(len(snap)))
            
        elif op == "RESTORE":
            name = cmd[1]
            store = {}
            if name in snapshots:
                for k, v in snapshots[name].items():
                    rem_ttl = v['remaining_ttl']
                    expiry = None if rem_ttl is None else current_time + rem_ttl
                    store[k] = {'value': v['value'], 'expiry': expiry}
            output.append("OK")

    return "\n".join(output)