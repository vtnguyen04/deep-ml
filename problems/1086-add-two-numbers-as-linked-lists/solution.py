def add_two_numbers(l1, l2):
    result = []
    carry = 0 
    
    i = 0
    len1, len2 = len(l1), len(l2)
    max_len = max(len1, len2)
    
    while i < max_len or carry > 0:
        val1 = l1[i] if i < len1 else 0
        val2 = l2[i] if i < len2 else 0
        
        total = val1 + val2 + carry
        
        carry = total // 10
        digit = total % 10
        
        result.append(digit)
        i += 1
        
    return result