using Makie, GeometryTypes

# Create a sphere with changing radius.
# Understand `lift` method

function setupCamera!(scene)
    cam = cam3d!(scene)
    eyepos = Makie.Vec3f0(5, 1.5, 0.5);
    lookat = Makie.Vec3f0(0., 0., 0.);
end

# ----------------------------------------------------------------------
marker_size = 0.0002
scene = Scene();
s1, radius = textslider(0.0f0:.1f0:0.5f0, "Radius", start = 0.1f0)
rad = Float32(to_value(radius))
Point3f0 = GeometryTypes.Point3f0

sphere = HyperSphere(Point3f0(0), 0.1f0)
positions = GeometryTypes.decompose(Point3f0, sphere)
#AP.meshscatter!(scene, positions, markersize=0.002, color=:blue, transparency=false)
AP.mesh(HyperSphere(Point3f0(0), 0.1f0), makersize=0.2, color=:blue)
AP.mesh!(scene, HyperSphere(Point3f0(0), 0.1f0), makersize=0.2, color=:blue)

parent_scene = Scene(resolution=(700, 700))
vbox(hbox(s1, scene), parent=parent_scene)

setupCamera!(parent_scene)
display(parent_scene)
