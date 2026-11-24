-- One month of raw TLC yellow trips, typed and flagged. No row is dropped here: each
-- exclusion is a flag, so the waterfall in quality_waterfall.sql always adds up.
create or replace temp view flagged as
with typed as (
    select
        "VendorID" as vendor_id,
        payment_type,
        "PULocationID" as pu_zone,
        "DOLocationID" as do_zone,
        tpep_pickup_datetime as picked_up_at,
        tpep_dropoff_datetime as dropped_off_at,
        trip_distance as miles,
        coalesce(fare_amount, 0) as fare,
        {cbd_fee} as cbd_fee,
        date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) / 3600.0 as hours
    from read_parquet('{path}')
)

select
    *,
    miles / nullif(hours, 0) as mph,
    case
        when not (picked_up_at >= date '{month}' and picked_up_at < date '{month}' + interval 1 month)
            then '1 pickup outside the file month'
        when pu_zone not between 1 and 263 or do_zone not between 1 and 263
            then '2 unknown or out-of-city zone'
        when fare <= 0 then '3 refund, void or zero fare'
        when payment_type not in (0, 1, 2) then '4 no charge, dispute or unknown'
        when hours <= 0 then '5 zero or negative duration'
        when hours < 1 / 60.0 or hours > 3 then '6 under 1 minute or over 3 hours'
        when miles < 0.2 or miles > 60 then '7 under 0.2 or over 60 miles'
        when miles / hours < 1 or miles / hours > 65 then '8 implausible speed'
        when payment_type = 0 then 'kept: flex fare (upfront price)'
        else 'kept: metered'
    end as status
from typed;
