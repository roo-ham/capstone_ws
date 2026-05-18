import os
from ament_index_python.packages import get_package_share_directory

# In your Node __init__:
pkg_share = get_package_share_directory('torque_controller')
urdf_path = os.path.join(pkg_share, 'urdf', 'hand_0926.urdf')