package main.models.distance;

import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

/**
 * Created by loongkuan on 26/03/16.
 **/

public abstract class Distance {
    WekaToSamoaInstanceConverter wekaConverter = new WekaToSamoaInstanceConverter();
    public abstract double findDistance(double[] p, double[] q);
}
