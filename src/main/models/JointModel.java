package main.models;

import main.models.posterior.PosteriorModel;
import main.models.prior.PriorModel;
import main.models.sampling.AbstractSampler;
import main.models.sampling.RandomSamples;
import weka.core.Debug;
import weka.core.Instances;

/**
 * Created by Lee on 30/01/2016.
 **/
public class JointModel extends AbstractModel{
    private PriorModel priorModel;
    private PosteriorModel posteriorModel;
    private AbstractSampler sampler;

    public JointModel(PriorModel priorModel, PosteriorModel posteriorModel, AbstractSampler sampler) {
        this.priorModel = (PriorModel) priorModel.copy();
        this.posteriorModel = (PosteriorModel) posteriorModel.copy();
        this.sampler = sampler.copy();
    }

    public JointModel(JointModel baseModel) {
        this.priorModel = (PriorModel) baseModel.priorModel.copy();
        this.posteriorModel = (PosteriorModel) baseModel.posteriorModel.copy();
        sampler = baseModel.sampler.copy();
    }

    public void setSampler(AbstractSampler sampler) {
        this.sampler = sampler;
        this.priorModel.setDataSet(sampler.getDataSet());
        this.posteriorModel.setDataSet(sampler.getDataSet());
    }

    public void setData(Instances data) {
        this.sampler.setDataSet(data);
        this.priorModel.setDataSet(data);
        this.posteriorModel.setDataSet(data);
    }

    public static double pyGvModelDistance(JointModel model1, JointModel model2) {
        Instances domain = findIntersectionBetweenInstances(model1.sampler.getSampledInstances(),
                model2.sampler.getSampledInstances());
        return model1.posteriorModel.findDistance(model1.posteriorModel, model2.posteriorModel, domain);
    }

    public static double pvModelDistance(JointModel model1, JointModel model2){
        RandomSamples sample = new RandomSamples((RandomSamples)model1.sampler, (RandomSamples)model2.sampler);
        return model1.priorModel.findDistance(model1.priorModel, model2.priorModel, sample)*sample.getMagnitudeScale();
    }

    @Override
    public void reset() {}

    @Override
    public JointModel copy() {
        return new JointModel(this);
    }
}
