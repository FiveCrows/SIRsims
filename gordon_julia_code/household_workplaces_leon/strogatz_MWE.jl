using LightGraphs, MetaGraphs

function ttest()
    sz = 400
    β = 0.4
    for i in 1:30
        if i > 20
            println("i= ", i)
            mgh = watts_strogatz(i, 20, β)
        end
    end
end

ttest()
println("mgh*= ", mgh)

println("mgh= $mgh")

function tst()
    for i in 1:10
        a = 3
    end
    print("a= ", a)
end

tst()
function tst()

    local a = 0

    for i in 1:10

        local b = 2

        a = 3

    end

    print("a= ", a)

end



tst()
