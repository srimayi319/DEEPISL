def isl_to_english_sentence(recognized_signs: list) -> str:
    """
    Takes a sequence of recognized ISL signs and applies rule-based logic
    to return a grammatically correct English sentence.
    """
    if not recognized_signs:
        return ""
        
    auxiliary_map = {
        'i': 'am',
        'he': 'is',
        'she': 'is',
        'we': 'are',
        'you': 'are',
        'they': 'are',
    }
    
    sentence_parts = []
    question_word = None
    
    # Process the list of signs
    temp_signs = [sign.lower() for sign in recognized_signs]
    
    # Find and remove question words to place at the beginning
    question_words = ['what', 'why', 'how', 'when', 'where']
    for q_word in question_words:
        if q_word in temp_signs:
            question_word = q_word
            temp_signs.remove(q_word)
            break
    
    # Handle specific patterns first
    if 'you' in temp_signs and 'name' in temp_signs:
        if question_word == 'what':
            return "What is your name?"
    
    if 'how' in temp_signs and 'you' in temp_signs:
        return "How are you?"
    
    # Reconstruct the sentence with auxiliary verbs
    i = 0
    while i < len(temp_signs):
        sign = temp_signs[i]
        next_sign = temp_signs[i+1] if i + 1 < len(temp_signs) else None

        # Rule: Insert auxiliary verb between subject and adjective/verb
        if sign in auxiliary_map and next_sign and next_sign in ['deaf', 'hearing', 'happy', 'sad', 'fine', 'tired', 'hungry']:
            sentence_parts.append(sign)
            sentence_parts.append(auxiliary_map[sign])
            sentence_parts.append(next_sign)
            i += 2
        else:
            sentence_parts.append(sign)
            i += 1
            
    # Format final sentence
    if sentence_parts:
        # Place question word at the beginning if present
        if question_word:
            sentence_parts.insert(0, question_word)
            
        final_sentence = " ".join(sentence_parts).capitalize()
        
        # Add appropriate punctuation
        if question_word:
            return final_sentence + '?'
        return final_sentence + '.'
    
    return ""