package main.models.posterior;

import main.models.AbstractModel;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by loongkuan on 1/04/16.
 **/
public abstract class PosteriorModel extends AbstractModel{
    public Instances dataSet;
    public abstract void setData(Instances data);
    public abstract double findPyGv(double classValue, Instance vector);
    public abstract double findDistance(PosteriorModel model1, PosteriorModel model2, Instances domain);
}
