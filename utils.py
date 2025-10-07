def isl_to_english_sentence(recognized_signs: list) -> str:
    """
    Enhanced NLP function to convert ISL signs to proper English sentences
    with better grammar and sentence structure.
    """
    if not recognized_signs:
        return ""
    
    # Convert to lowercase and clean
    temp_signs = [str(sign).lower().strip() for sign in recognized_signs if str(sign).strip()]
    
    if not temp_signs:
        return ""
    
    # Grammar rules and transformations
    subject_verbs = {
        'i': 'am',
        'he': 'is', 
        'she': 'is',
        'it': 'is',
        'we': 'are',
        'you': 'are',
        'they': 'are',
        'this': 'is',
        'that': 'is'
    }
    
    question_words = {'what', 'why', 'how', 'when', 'where', 'who', 'which'}
    helping_verbs = {'am', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'have', 'has', 'can', 'will'}
    
    # Common ISL patterns to English conversions
    pattern_replacements = {
        ('you', 'name', 'what'): 'What is your name?',
        ('how', 'you'): 'How are you?',
        ('what', 'your', 'name'): 'What is your name?',
        ('my', 'name'): 'My name is',
        ('i', 'am', 'fine'): 'I am fine',
        ('thank', 'you'): 'Thank you',
        ('you', 'how'): 'How are you?',
        ('what', 'this'): 'What is this?',
        ('where', 'you', 'from'): 'Where are you from?',
    }
    
    # Check for exact pattern matches first
    signs_tuple = tuple(temp_signs)
    for pattern, replacement in pattern_replacements.items():
        if len(signs_tuple) >= len(pattern) and signs_tuple[:len(pattern)] == pattern:
            return replacement
    
    # Process sentence structure
    processed_words = []
    i = 0
    
    while i < len(temp_signs):
        current_word = temp_signs[i]
        next_word = temp_signs[i + 1] if i + 1 < len(temp_signs) else None
        prev_word = temp_signs[i - 1] if i > 0 else None
        
        # Handle subject-verb agreement
        if current_word in subject_verbs and next_word and next_word not in helping_verbs:
            if next_word not in question_words:  # Don't insert verb before question words
                processed_words.append(current_word)
                processed_words.append(subject_verbs[current_word])
                i += 1
                continue
        
        # Handle questions
        if current_word in question_words:
            # Question word goes to the beginning
            if processed_words and processed_words[0] not in question_words:
                processed_words.insert(0, current_word)
            else:
                processed_words.append(current_word)
            i += 1
            continue
        
        # Handle possessives
        if current_word == 'your' and next_word == 'name':
            processed_words.extend(['what', 'is', 'your', 'name'])
            i += 2
            continue
            
        if current_word == 'my' and next_word == 'name':
            processed_words.extend(['my', 'name', 'is'])
            i += 2
            continue
        
        # Basic word order corrections
        if current_word == 'you' and next_word == 'how':
            processed_words.extend(['how', 'are', 'you'])
            i += 2
            continue
            
        if current_word == 'name' and prev_word == 'your':
            # Already handled above
            i += 1
            continue
        
        # Default: just add the word
        processed_words.append(current_word)
        i += 1
    
    # Final sentence cleaning and formatting
    if not processed_words:
        return ""
    
    # Remove consecutive duplicates but preserve meaning
    final_words = []
    for word in processed_words:
        if not final_words or word != final_words[-1]:
            final_words.append(word)
    
    # Capitalize first word
    if final_words:
        final_words[0] = final_words[0].capitalize()
    
    # Add punctuation
    sentence = " ".join(final_words)
    
    # Determine punctuation
    has_question_word = any(word in final_words for word in question_words)
    is_question_pattern = (
        has_question_word or 
        sentence.lower().startswith(('how', 'what', 'why', 'when', 'where', 'who', 'which'))
    )
    
    if is_question_pattern:
        sentence = sentence + '?'
    else:
        sentence = sentence + '.'
    
    # Final cleanup of common issues
    sentence = sentence.replace(' ?', '?').replace(' .', '.')
    sentence = sentence.replace('am are', 'am')  # Fix double verbs
    sentence = sentence.replace('is are', 'is')
    sentence = sentence.replace('  ', ' ')  # Remove double spaces
    
    return sentence

# Additional utility functions
def smooth_predictions(predictions, window_size=3):
    """Apply smoothing to prediction sequence"""
    if len(predictions) < window_size:
        return predictions[-1] if predictions else None
    
    # Return the most common prediction in the window
    recent = predictions[-window_size:]
    return max(set(recent), key=recent.count)

def calculate_confidence_metrics(confidence_scores):
    """Calculate confidence metrics for predictions"""
    if not confidence_scores:
        return 0.0
    
    recent_scores = confidence_scores[-5:]  # Last 5 predictions
    return sum(recent_scores) / len(recent_scores)