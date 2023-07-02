# RL with Differentiable Rigid 3D env

TLDR: differentiable simulation using nimblephysics and visualize using sapien

Notes update:
1. some nimble methods will segfault for unknown reason

#### Overview of main components
* rigid3d_simulator.py
    this class manages everything about simulation and rendering
    1. manages simulation using nimble physics
    2. render or create gui using sapien viewer
    3. serve nimble viewer server at localhost
* sapien_viewer.py
    operates sapien as a viewer for nimblephysics, to make debugging easier this by default uses the same camera angle as nimble visualizer
* r3d_base.py
    the base class for RL environment, inherited from GoalEnv and implements
* nimble_ref_frame.png
    this is the default reference frame visualized in default camera angle (both nimble and sapien viewer)
* gen_nimble_doc.py
    1. doc website at https://nimblephysics.org/
    2. this script writes help(object) of frequently used classes to nimble_help_docs

#### Overview of tests and things to be careful. For more details please run the demo directly to watch the video in tests folder
* test_box_roll.py
    demo of box fall on ground
    1. tune urdf inertia matrix if the body does not rotate easily
* test_gradient.py
    demo of correctness of nimble collision gradient
* test_ik_jaco2.py
    demo of jaco2 pushing a box using IK control
    1. objects: jaco2.urdf, ground.urdf, box.urdf
    2. simulation statistics (rendering time is not considered)
        a. time used per step (no collision): 0.06 s
        b. time used per step (collision): 0.06-0.07 s
    3. **restitution_coeff must be set for correct collision behavior**
        this is currently managed as an argument of the load_urdf function in rigid3d_simulator class, setting this value to 0 also results in behaviors like object clip through each other
* test_ik_simple.py
    demo of a simple 5 box link robot pushing a box using IK control
    1. objects: beginner.urdf, ground.urdf, box.urdf
    2. simulation statistics (rendering time is not considered)
        a. time used per step (no collision): 0.0002 s
        b. time used per step (collision): 0.0002-0.0003 s
    this is 300 times faster than having a mesh collision shaped robot arm like jaco2
* test_urdf.py
    a helper script for building urdf xml and check it in sapien and nimble
    1. set path to target urdf, then run this script
    2. this will detect changes to this urdf, and update to sapien viewer and nimblephysics by reloading this loaded robot
* test_vec_env.py
    this does not work, nimble does not support multiple instances in the same process.
