"""
This is a reusable library containing useful decorators and functions including:
    -- Custom timing
    -- Showing an image using OpenCV

History:

v1.0: Created by Erik Thomas -- September 2018
v1.1: Updated by Erik Thomas -- December 2018
v1.2: Updated by Erik Thomas -- January 2019
"""

def timer(func_in):
    """
    Decorator used to print the length of time taken for a function to execute
    
    Keyword arguments:
    func_in -- the function object to be timed
    """
    import time
    
    def time_calc(*args, **kw):
        start_time = time.time()
        result = func_in(*args, **kw)
        end_time = time.time()
        print(f"Function: {func_in.__name__} took {end_time-start_time} seconds to execute")
        return result
    return time_calc

def opencv_show_image(image, window_name="default"):
    """
    Shows a given image (numpy array). Locks up the program until a key is pressed
    
    Keyword arguments:
    image -- the image to be displayed
    window_name -- the display name for the window created by cv2.imshow()
    """
    try:
        import cv2
    except ModuleNotFoundError as module_err:
        print(f"Issue finding OpenCV: {module_err}")
        return
    
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def reset_file(file_path: str, headings: str = ""):
    """
    Used to clear a file after it has been appended to
    
    Keyword arguments:
    file_path -- the file to be reset
    headings -- the top line to use for the file (if a csv file)
    """
    with open(file_path, 'w') as file_to_reset:
        file_to_reset.write(headings)
        file_to_reset.write('\n')
        
def write_to_image(image: list, text: str, colour: list = (255, 255, 255), 
                   position: list = (0, 0), width: int = 4):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, width, colour, 2, cv2.LINE_AA)
    return image

"""
TESTING GROUND -- THESE FUNCTIONS ARE NOT READY FOR USE YET
"""

class InvalidRaiseException(Exception):
    pass

def throws(ErrorType):
    from functools import wraps
    def decorator(func_in):
        @wraps(func_in)
        def wrapped(*args, **kwargs):
            try:
                return func_in(*args, **kwargs)
            except ErrorType:
                raise
            except InvalidRaiseException:
                raise
            except Exception as err:
                raise InvalidRaiseException(f"Got {err.__class__}, expected {err.__name__}, from {err.__name__}")
        return wrapped
    return decorator