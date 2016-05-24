package main.models.prior;

import main.models.AbstractModel;
import main.models.sampling.AbstractSampler;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;

/**
 * Created by loongkuan on 1/04/16.
 */
public interface PriorModel {
    void setDataSet(Instances dataSet);
    void setSampler(AbstractSampler sampler);
    double findPv(double[] vector);
    double findDistance(PriorModel model1, PriorModel model2, AbstractSampler sample);
    PriorModel copy();
}
