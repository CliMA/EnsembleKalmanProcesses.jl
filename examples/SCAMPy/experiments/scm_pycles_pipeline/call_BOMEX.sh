# Allows use of tr in Mac
export LC_CTYPE=C
# Recover name of SCAMPy directory
scm_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Simulation names
declare -a simnames=("Bomex")
# Values to be modified
val_prev_param=0.01
uuid_prev=51515

num_params=$#/2
suffix=""
allparams=""
for (( i=1; i<=$num_params; i++ ))
do
    j=$((num_params+i))
    suffix=${suffix}_${!i}
    allparams=${allparams}${!i}
    echo ${!j}
    echo ${!i}
done

# Loop over simulations
for simname in "${simnames[@]}"
do
    # Create directory to store input files
    mkdir sim_${simname}${suffix}
    cp Output.${simname}.00000/paramlist_${simname}.in sim_${simname}${suffix}/paramlist_${simname}.in
    cp Output.${simname}.00000/${simname}.in sim_${simname}${suffix}/${simname}.in
    cp Output.${simname}.00000/paramlist_${simname}.in paramlist_${simname}.in
    cp Output.${simname}.00000/${simname}.in ${simname}.in

    # Modify parameters from input files
    for (( i=1; i<=$num_params; i++ ))
    do
        j=$((num_params+i))
        line_param=$( awk '/"'${!j}'/{print NR}' sim_${simname}${suffix}/paramlist_${simname}.in )
        gawk  'NR=='$line_param'{gsub(/'$val_prev_param'/,'"${!i}"')};1' sim_${simname}${suffix}/paramlist_${simname}.in > tmp${suffix} && mv tmp${suffix} sim_${simname}${suffix}/paramlist_${simname}.in
    done

    # Generate random 5-digit number to use as UUID
    uuid=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 5 | head -n 1)
    line_uuid=$( awk '/"uuid/{print NR}' sim_${simname}${suffix}/${simname}.in )
    awk 'NR==row_num {sub(val,val2)};1' row_num="$line_uuid" val="$uuid_prev" "val2=$uuid" sim_${simname}${suffix}/${simname}.in > tmp_${uuid} && mv tmp_${uuid} sim_${simname}${suffix}/${simname}.in

    output_dir=$(awk -F"output_root" '/output_root/{print $2}' sim_${simname}${suffix}/${simname}.in)
    output_dir=$(echo "$output_dir" | sed 's|[": ]||g')

    # Run SCAMPy with modified parameters
    python ${scm_dir}/main.py sim_${simname}${suffix}/${simname}.in sim_${simname}${suffix}/paramlist_${simname}.in
    # Copy used input files to output directory, since the copied files by SCAMPy are not the ones that are used.
    cp sim_${simname}${suffix}/paramlist_${simname}.in ${output_dir}Output.${simname}.${uuid}/paramlist_${simname}.in
    cp sim_${simname}${suffix}/${simname}.in ${output_dir}Output.${simname}.${uuid}/${simname}.in
    rm -r sim_${simname}${suffix}
    rm paramlist_${simname}.in ${simname}.in
    echo ${output_dir}Output.${simname}.${uuid} >> ${allparams}.txt
done

