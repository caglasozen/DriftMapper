package main.models.distance;

import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import main.models.ClassifierModel;
import moa.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.HashMap;
import java.util.stream.DoubleStream;

/**
 * Created by loongkuan on 26/03/16.
 **/

public abstract class Distance {
    WekaToSamoaInstanceConverter wekaConverter = new WekaToSamoaInstanceConverter();

    public abstract double findPyGvDistance(ClassifierModel model1, ClassifierModel  model2, Instances allInstances);
    public abstract double findPvDistance(ClassifierModel  model1, ClassifierModel  model2, Instances allInstances);

}
