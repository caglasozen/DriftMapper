package main.models.prior;

import main.models.AbstractModel;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;

/**
 * Created by loongkuan on 1/04/16.
 */
public abstract class PriorModel extends AbstractModel {
    public Instances allPossibleInstances;
    Instances dataSet;
    ArrayList<ArrayList<String>> xPossibleValues;

    public abstract void setDataSet(Instances dataSet);
    public abstract double findPv(double[] vector);
    public abstract double findDistance(PriorModel model1, PriorModel model2, Instances domain);

}
