#utils for reading in parameters and writing parameters using toml

function find_in_tags(tags_string,string_list)
    #assume comma separation, this trims whitespace
    tags_list=strip.(split(tags_string,","))
    #returns true only if everything in stringlist occurs somewhere in tags_list
    return all(i in tags_list for i in string_list)
end

function get_tagged_parameters(model_tags,toml_dict)
    parameter_dict = Dict()

    # function returning true if all model_tags are in the current parameter 
    for (key,val) in toml_dict
        if haskey(val,"Tags")
            if find_in_tags(val["Tags"],model_tags)
                if val["Prior"] != "fixed"
                    #prefix = component*"."*parameterization*"."
                    parameter_dict[key] = Dict("Tags"           => val["Tags"],
                                               "RunValue"       => val["RunValue"],
                                               "Prior"          => val["Prior"],
                                               "Transformation" => val["Transformation"])
                end
            end
        end
    end
    return parameter_dict
end

function create_dict_from_ensemble(parameter_dict, initial_params, param_names)
    param_dict_ensemble=[]
    for param_set in eachrow(initial_params') # for each parameter realization
        #overwrite the param set
        for (idx,name) in enumerate(param_names)
            parameter_dict[name]["RunValue"] = param_set[idx]
        end
        push!(param_dict_ensemble,deepcopy(parameter_dict))
    end
    return param_dict_ensemble
end

function write_toml_ensemble_verbatim(ens_dir_name,filename,param_dict_ensemble)
    if ~isdir(ens_dir_name)
        mkdir(ens_dir_name)
    else
        println("overwriting files in ", ens_dir_name)
        rm(ens_dir_name,recursive=true)
        mkdir(ens_dir_name)
        
    end
    for (idx,param_dict) in enumerate(param_dict_ensemble)
        member_filename = ens_dir_name*"/member_"*string(idx)*".toml" #toml file
        io_member_file=open(member_filename, "a")
        param_names = [ "["*string(key)*"]" for (key,val) in param_dict]
        param_vals = ["RunValue = "*string(param_dict[key]["RunValue"])*"\n" for (key,val) in param_dict]
        open(filename) do io
            #gets the right toml name
            change_value=[Int(0)]
            change_value_flag=[false]
            for line in eachline(io, keep=true)
                for (name_idx,name) in enumerate(param_names)
                    if contains(line, name) #found the parameter
                        # change the next RunValue instance
                        change_value[1] = name_idx
                        change_value_flag[1] = true
                    end
                    # if we have flagged a parameter value to be changed, we change it and then remove the flag.
                    if change_value_flag[1] == true
                        if contains(line, "RunValue") #found the parameter
                            line = param_vals[change_value[1]]
                            change_value_flag[1]=false
                        end
                    end
                end
                write(io_member_file,line) #write the line to new file
            end
            
        end
        close(io_member_file)
    end
    
end

function write_toml_ensemble_from_dict(ens_dir_name,toml_dict,param_dict_ensemble)
    if ~isdir(ens_dir_name)
        mkdir(ens_dir_name)
    else
        println("overwriting files in ", ens_dir_name)
        rm(ens_dir_name,recursive=true)
        mkdir(ens_dir_name)
    end

    
    for (idx,param_dict) in enumerate(param_dict_ensemble)
        member_filename = ens_dir_name*"/member_"*string(idx)*".toml" #toml file
        io_member_file=open(member_filename, "a")
        param_names = [ "["*string(key)*"]" for (key,val) in param_dict]
        param_vals = ["RunValue = "*string(param_dict[key]["RunValue"])*"\n" for (key,val) in param_dict]

        #for each parameter that needs changing. The toml key is the same as the param_dict key
        for (key,val) in param_dict #key in param_dict, is the same as in toml dict
            toml_dict[key]["RunValue"] = val["RunValue"] 
        end
        
        open(member_filename,"w") do io
            TOML.print(io,toml_dict)
        end
        
    end      
    
end

# Don't need this typically... as we have the ekobject,
# also this will create a bunch of prior distribution info
function read_ensemble_from_dir(ens_dir_name)
    filename_ensemble = ens_dir_name * "/" .* filter(contains(".toml"), readdir(ens_dir_name))
    N_ens = filename_ensemble
    toml_dict_dict = Dict()
    parameter_dict_dict = Dict()
    for filename in filename_ensemble
        split(split(filename,"_")[2],".")[1]
        toml_dict_dict[filename] = TOML.parsefile(filename)
        parameter_dict_dict[filename] = get_tagged_parameters(model_tags, toml_dict_dict[filename])
    end
    return toml_dict_dict, parameter_dict_dict, N_ens
end
