package main.models.prior;

import main.models.AbstractModel;
import weka.core.Instances;

/**
 * Created by loongkuan on 1/04/16.
 */
public abstract class PriorModel extends AbstractModel {
    public Instances allPossibleInstances;
    public abstract void setData(Instances data);
    public abstract double findPv(double[] vector);
    public abstract double findDistance(PriorModel model1, PriorModel model2, Instances domain);
}
