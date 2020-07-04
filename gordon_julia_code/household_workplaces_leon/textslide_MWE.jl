using Makie

scene = Scene()
s1, radius = textslider(0.0f0:0.1f0:1.0f0, "Radius", start = 0.2f0)

textslider(0:2π/32:2π, "ϕ", start=π/4, sliderheight=100, sliderlength=800)
textslider(0:2π/32:2π, "ϕ", start=π/4)
textslider(0:2π/32:2π, "radius", start=π/4)
display(scene)
