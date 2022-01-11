import Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")

using CSV
using DataFrames
using Random

function read_csv(name="")
    csv_render = CSV.File(name)
    data = DataFrame(csv_render)
    return data
end



function entropy(dataset)
    len = size(dataset)[1]

    if len == 0
        return 0
    end
    
    attribute_list = names(dataset)

    target_attribute_name = attribute_list[length(attribute_list)]
    target_attribute_values = unique(dataset[:, target_attribute_name])
    entropy = 0

    for value in target_attribute_values
        p = count(x->x == value, dataset[:, target_attribute_name])/len
        entropy += -p * log2(p)
    end

    return entropy
end

function info_gain(attribute_name, split, dataset)
    set_smaller = filter(attribute_name => <(split), dataset)
    p_smaller = size(set_smaller)[1] / size(dataset)[1]
    set_greater_equals = filter(attribute_name => <=(split), dataset)
    p_greater_equals = size(set_greater_equals)[1] / size(dataset)[1]

    info_gain = entropy(dataset)
    info_gain += -p_smaller * entropy(set_smaller)
    info_gain += -p_greater_equals * entropy(set_greater_equals)

    return info_gain
end

function max_info_gain(attribute_list, dataset)
    max_info_gain = 0
    max_info_gain_attribute = ""
    max_info_gain_split = 0

    for attribute in attribute_list
        sort!(dataset, attribute)
        for split in dataset[:, attribute]
            split_info_gain = info_gain(attribute, split, dataset)
            if split_info_gain >= max_info_gain
                max_info_gain = split_info_gain
                max_info_gain_attribute = attribute
                max_info_gain_split = split
            end
        end
    end
    return max_info_gain, max_info_gain_attribute, max_info_gain_split
end

mutable struct Node
    isLeaf :: Bool
    depth :: Int
    G :: Float64
    attribute :: String
    split :: Float64
    dataset :: DataFrame
    majority :: String
    left :: Any
    right :: Any

    function Node(_dataset::DataFrame, _depth::Int)
        new(false, _depth, entropy(_dataset), "", 0, _dataset, "", nothing, nothing)
    end
end

function get_majority(dataset)
    max = 1
    attribute_list = names(dataset)
    target_attribute_name = attribute_list[length(attribute_list)]
    groups = groupby(dataset, target_attribute_name)
    for i in 1:size(groups)[1]
        if size(groups[i])[1] > size(groups[max])[1]
            max = i
        end
    end
    return groupby(dataset, target_attribute_name)[max][1, length(attribute_list)]
end

function build(node::Node)
    node.majority = get_majority(node.dataset)
    if node.G == 0 
        node.isLeaf = true
        return
    end
    if node.depth == MAX_HEIGHT - 1
        node.isLeaf = true
        return
    end
    
    attribute_list = names(node.dataset)
    attribute_list = deleteat!(attribute_list, length(attribute_list))
    node.attribute = max_info_gain(attribute_list, node.dataset)[2]
    node.split = max_info_gain(attribute_list, node.dataset)[3]
    set_smaller_equals = filter(node.attribute => <=(node.split), node.dataset)
    set_greater = filter(node.attribute => >(node.split), node.dataset)

    node.left = Node(set_smaller_equals, node.depth + 1)
    node.right = Node(set_greater, node.depth + 1)

    build(node.left)
    build(node.right)
end

function print_tree(root::Node)
    if root === nothing
        return
    end
    if root.isLeaf
        println(root.majority)
        return
    end
    println(root.attribute, "<=", root.split, "?")
    print("\t"^(root.depth + 1), "[True]") 
    print_tree(root.left)
    print("\t"^(root.depth + 1), "[False]")
    print_tree(root.right)
end

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

function test(root::Node, dataset)
    variety = dataset[1, "variety"]
    temp = root
    while temp.isLeaf == false
        if dataset[1, temp.attribute] <= temp.split
            temp = temp.left
        else
            temp = temp.right
        end
    end
    if variety == temp.majority
        return true
    end

    return false
end

function calculate_accuracy(root::Node, dataset)
    len = size(dataset)[1]

    accuracy = 0

    for i in 1:len
        if test(root, DataFrame(dataset[i, :]))
            accuracy += 1/len
        end
    end

    return accuracy
    
end

data = read_csv("iris.csv")
sets = splitdf(data, 2/3)
training_set = sets[1]
test_set = sets[2]

MAX_HEIGHT = 4
root = Node(DataFrame(training_set), 0)
build(root)


accuracy = calculate_accuracy(root, DataFrame(test_set))*100
println("Decision tree:")
print_tree(root)
println("Accuracy: ", accuracy, "%")
