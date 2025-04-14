ASE=NSSP534frc2esm_f19_tn14_BECCS_land_190124 #BECCS-land
#CASEdir=/projects/NS9576K/anusha/noresm/cases/BECCS/$CASE/lnd/hist
 
CASE=NSSP534frc2esm_f19_tn14_20230106  #BECCS_baseline
CASEdir=/projects/NS9560K/noresm/cases/$CASE/lnd/hist
 
echo "now at `pwd`"
# List of variables to process
#variables=("TOTVEGC" "GPP" "NBP" "TOTSOMC")
variables=("NPP" "AGNPP" "EFLX_LH_TOT" "ER")
 
# Loop through each variable
for var in "${variables[@]}"; do
    for file in $CASEdir/$CASE.clm2.h0.*.nc; do
        # Extract the variable from each file and save to separate files
        /nird/services/software/nird/cdo selvar,$var "$file" "${var}-h0_${file##*/}"
    done
 
    # Concatenate all variable files into a single file
    /nird/services/software/nird/cdo cat ${var}-h0_*.nc ${var}_${CASE}.clm_h0_2015-2100.nc
    # Clean up intermediate files
    rm -rf ${var}-h0_*.nc
    /nird/services/software/nird/cdo select,startdate=2015-01-01,enddate=2030-01-01 ${var}_${CASE}.clm_h0_2015-2100.nc ${var}_${CASE}.clm_h0_2015-2029_cropped.nc          #create a 2015 to 2030 file
 
    # Calculate annual means
    /nird/services/software/nird/cdo shifttime,-1month ${var}_${CASE}.clm_h0_2015-2100.nc ${var}_${CASE}.clm_h0_2015-2100_timecorrctd.nc
    /nird/services/software/nird/cdo yearmean ${var}_${CASE}.clm_h0_2015-2100_timecorrctd.nc annualmean_${var}_${CASE}.clm_h0_2015-2100.nc
 
    #/nird/services/software/nird/cdo selyear,2015/2029 annualmean_${var}_${CASE}.clm_h0_2015-2100.nc annualmean_${var}_${CASE}.clm_h0_2015-2029_cropped.nc #create a 2015 to 2030 file
done
