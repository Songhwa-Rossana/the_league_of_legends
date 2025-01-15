import numpy as np

def palm_rots(palms_landmarks)-> np.array:
    """Compute the palm rotation and slope towards cammera
       Args:
       -----
       palms_landmarks: array
         (n_palms, n_landmarks, 2)

       Returns: array
         (n_palms, )
    """
    l_index = palms_landmarks[:, 4]  # Landmark corresponding to the index finger
    l_little = palms_landmarks[:, 1] # Landmark corresponding to little finger
    l_last = palms_landmarks[:, -1]  # Last landmark

    v_little_index = l_little - l_index
    v_little_index[:, 0] = np.abs(v_little_index[:, 0]) # Take vector always in positive magnitude

    # Palm rotation: Rotation of vector l_little, l_index (2D)
    palm_rot = np.arctan2(v_little_index[:, 1], v_little_index[:, 0])

    """# Palm width: the distance between index and little finger
    palm_width =  np.sqrt(np.sum(v_little_index ** 2, axis=1))

    # Slope: Distance of last landmark to previous vector
    # Equation of line through (l_index, l_little)
    # We use the equation of the line in the form Ax + By + C = 0
    A = v_little_index[1]
    B = -v_little_index[0]
    C = l_index[1] * v_little_index[0] - l_index[0] * v_little_index[1]

    # Calculate the distance of the last landmark to this line (point-to-line distance)
    d = np.abs(A * l_last[0] + B * l_last[1] + C) / palm_width #np.sqrt(A**2 + B**2)

    # Slope: normalized distance from palm plane to the last landmark
    slope_r = d / palm_width"""

    return palm_rot

def get_palm_params(palm):

    """Retrieve the palm landmarks, the bbox containing a palm, and the palm rot
        from the raw palm parameters
    """

    palm_landmarks = palm[4:-2].astype(np.int32).reshape(-1, 2)
    palm_bbox = [palm_landmarks[:, 0].min(),
              palm_landmarks[:, 1].min(),
              palm_landmarks[:, 0].max(),
              palm_landmarks[:, 1].max(),] # [x1, y1, x2, y2]
    palm_rot = palm[-1]

    return palm_bbox, palm_landmarks, palm_rot

def prompt_text_user(prompt: str):

  """Creates a terminal prompt to tell the user to enter a text
    Args:
    ----
    prompt: str
      Prompt to show the user
  """
  print(f'{prompt}: ', end=' ')
  input_data = input()
  return input_data