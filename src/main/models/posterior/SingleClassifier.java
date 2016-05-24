package main.models.posterior;

import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import main.models.AbstractModel;
import main.models.distance.Distance;
import main.models.distance.HellingerDistance;
import main.models.distance.TotalVariation;
import main.models.sampling.AbstractSampler;
import main.models.sampling.AllSamples;
import moa.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.stream.DoubleStream;

/**
 * Created by loongkuan on 1/04/16.
**/
public class SingleClassifier extends AbstractModel implements PosteriorModel{
    private HashMap<Double, Integer> classValueToIndex = new HashMap<>();
    private WekaToSamoaInstanceConverter wekaConverter = new WekaToSamoaInstanceConverter();
    private Classifier baseClassifier;

    public SingleClassifier(Classifier baseClassifier, Instances dataSet) {
        this.sampler = new AllSamples(dataSet);
        this.baseClassifier = baseClassifier;
    }

    public SingleClassifier(Classifier baseClassifier, AbstractSampler sampler) {
        this.sampler = sampler;
        this.baseClassifier = baseClassifier;
    }

    private void trainClassifier() {
        baseClassifier.resetLearning();
        baseClassifier.prepareForUse();
        for (Instance inst : this.sampler.getDataSet()) {
            baseClassifier.trainOnInstance(wekaConverter.samoaInstance(inst));
        }
    }

    private void getClassLabels() {
        for (Instance inst : this.sampler.getDataSet()) {
            classValueToIndex.put(inst.classValue(), (int)inst.classValue());
        }
    }

    @Override
    public void setDataSet(Instances dataSet) {
        this.sampler.setDataSet(dataSet);
        reset();
    }

    @Override
    public void setSampler(AbstractSampler sampler) {
        this.sampler = sampler;
        reset();
    }

    @Override
    public SingleClassifier copy() {
        Classifier newClassifier = baseClassifier.copy();
        return new SingleClassifier(newClassifier, this.sampler);
    }

    @Override
    public void reset() {
        trainClassifier();
        getClassLabels();
    }

    @Override
    public double findPyGv(double classValue, Instance vector) {
        // Check if class given is in data set
        if (!this.classValueToIndex.containsKey(classValue)) return 0.0f;
        // Get classifier's votes for instance/vector
        double[] votesForInstance = baseClassifier.getVotesForInstance(wekaConverter.samoaInstance(vector));
        // Normalise votes
        double sumVotes = DoubleStream.of(votesForInstance).sum();
        double[] normVotes = new double[votesForInstance.length];
        for (int i = 0; i < votesForInstance.length; i++) {
            normVotes[i] = (sumVotes == 0.0) ? 0.0f : votesForInstance[i] / sumVotes;
        }
        return normVotes[this.classValueToIndex.get(classValue)];
    }

    @Override
    public double findDistance(PosteriorModel model1, PosteriorModel model2, Instances domain) {
        Double[] classValues = this.classValueToIndex.keySet().toArray(new Double[classValueToIndex.size()]);
        Distance distanceMetric = new TotalVariation();
        double driftMag = 0.0f;

        for (Instance inst : domain) {
            double[] p = new double[classValues.length];
            double[] q = new double[classValues.length];
            for (int k = 0; k < classValues.length; k++) {
                p[k] = model1.findPyGv(classValues[k], inst);
                q[k] = model2.findPyGv(classValues[k], inst);
            }
            driftMag += distanceMetric.findDistance(p, q);
        }
        driftMag = driftMag / domain.size();
        return driftMag;
    }
}
