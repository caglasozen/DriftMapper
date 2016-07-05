package main.models;

import main.models.distance.Distance;
import main.models.distance.TotalVariation;
import main.models.posterior.PosteriorModel;
import main.models.prior.PriorModel;
import main.models.sampling.AbstractSampler;
import main.models.sampling.AllSamples;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by loongkuan on 11/05/16.
 **/
public class NaiveMatrix extends AbstractModel implements PriorModel, PosteriorModel{
    private int[][] combinationFreq;

    private double[][] max;
    private double[][] min;
    private double[] mean;
    private double[] sd;

    public NaiveMatrix(Instances dataSet) {
        this.setDataSet(dataSet);
        this.reset();
    }

    public NaiveMatrix(AbstractSampler sampler) {
        this.setSampler(sampler);
        this.reset();
    }

    @Override
    public void reset() {
        int nClasses = this.sampler.getAllClasses().size();

        int nCombinations = 1;
        ArrayList<ArrayList<String>> attributesValues = this.sampler.getAllPossibleValues();
        for (ArrayList<String> attributeValue : attributesValues) {
            nCombinations *= attributeValue.size();
        }
        combinationFreq = new int[nCombinations][nClasses + 1];
        for (Instance inst : this.sampler.getDataSet()) {
            double[] instValues = Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1);
            int hash = this.convertInstToHash(instValues);
            int classHash = this.convertClassToHash(inst.toDoubleArray()[inst.numAttributes() - 1]);
            this.combinationFreq[hash][0] += 1;
            this.combinationFreq[hash][classHash] += 1;
        }

        // Reset statistics
        this.max = new double[2][3];
        this.min = new double[2][3];
        this.mean = new double[2];
        this.sd = new double[2];
        for (int i = 0; i< 2; i ++) {
            this.max[i] = new double[sampler.getAllPossibleValues().size() + 1];
            this.max[i][0] = -1.0f;
            this.min[i] = new double[sampler.getAllPossibleValues().size() + 1];
            this.min[i][0] = 2.0f;
            for (int j = 1; j < max[i].length; j++) {
                this.max[i][1] = -1.0;
                this.min[i][1] = -1.0;
            }
            this.mean[i] = 0.0f;
            this.sd[i] = 0.0f;
        }

    }

    @Override
    public NaiveMatrix copy() {
        return new NaiveMatrix(this.sampler);
    }

    @Override
    public void setDataSet(Instances dataSet) {
        this.sampler = new AllSamples(dataSet);
    }

    @Override
    public void setSampler(AbstractSampler sampler) {
        this.sampler = new AllSamples(sampler.getDataSet());
    }

    @Override
    public double findPv(double[] vector) {
        return this.vectorExists(vector) ?
                (double)this.combinationFreq[this.convertInstToHash(vector)][0]/
                        (double)this.sampler.getDataSet().size() : 0.0f;
    }

    public double findPy(double classValue) {
        int classIndex = this.convertClassToHash(classValue);
        if (classIndex == -1) return 0.0f;
        int nOccurrences = 0;
        for (int i = 0; i < this.combinationFreq.length; i++) {
            nOccurrences += this.combinationFreq[i][classIndex];
        }
        return (double)nOccurrences / (double)this.sampler.getDataSet().size();
    }

    @Override
    public double findPyGv(double classValue, Instance vector) {
        double[] dVector = Arrays.copyOfRange(vector.toDoubleArray(), 0, vector.numAttributes() - 1);
        int classIndex = this.convertClassToHash(vector.toDoubleArray()[vector.numAttributes() - 1]);
        return this.vectorExists(dVector) && classIndex != -1 ?
                (double)this.combinationFreq[this.convertInstToHash(dVector)][1+classIndex]/
                        (double)this.combinationFreq[this.convertInstToHash(dVector)][0]: 0.0f;
    }

    @Override
    public double findDistance(PriorModel model1, PriorModel model2, AbstractSampler sample) {
        // Trim last attribute as allPossibleCombinations contains a class attribute wh
        Distance distanceMetric = new TotalVariation();

        double[] p = new double[sample.getNInstances()];
        double[] q = new double[sample.getNInstances()];
        double[] dist = new double[sample.getNInstances()];
        sample.reset();
        for (int i = 0; i < sample.getNInstances(); i++) {
            Instance inst = sample.nextInstance();
            p[i] = model1.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1));
            q[i] = model2.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1));
            dist[i] = distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
            if (dist[i] > max[0][0]) max[0] = ArrayUtils.addAll(new double[]{dist[i]}, inst.toDoubleArray());
            if (dist[i] < min[0][0]) min[0] = ArrayUtils.addAll(new double[]{dist[i]}, inst.toDoubleArray());
        }
        double finalDist = distanceMetric.findDistance(p, q);
        this.mean[0] = 0.0f;
        for (int i = 0; i < sample.getNInstances(); i++) {
            mean[0] += dist[i];
        }
        mean[0] /= sample.getNInstances();
        this.sd[0] = 0.0f;
        for (int i = 0; i < sample.getNInstances(); i++) {
            sd[0] += Math.pow((dist[i] - mean[0]), 2);
        }
        sd[0] /= sample.getNInstances() - 1;
        sd[0] = Math.sqrt(sd[0]);
        return finalDist;
    }

    @Override
    public double findDistance(PosteriorModel model1, PosteriorModel model2, Instances sample) {
        return 0.0f;
    }

    public double[] findDistance(NaiveMatrix model1, NaiveMatrix model2, AbstractSampler sample) {
        Distance distanceMetric = new TotalVariation();
        int nClasses = sample.getAllClasses().size();
        double finalDist = 0.0f;

        sample.reset();
        for (int i = 0; i < sample.getNInstances(); i++) {
            Instance inst = sample.nextInstance();
            double[] p = new double[nClasses];
            double[] q = new double[nClasses];
            double[] dist = new double[nClasses];
            double weight = model1.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1)) +
                    model2.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1));
            for (int j = 0; j < nClasses; j++) {
                double yClass = Double.parseDouble(sample.getAllClasses().get(j));
                p[j] = model1.findPyGv(yClass, inst);
                q[j] = model2.findPyGv(yClass, inst);
                dist[j] = distanceMetric.findDistance(new double[]{p[j]}, new double[]{q[j]}) * weight;
                if (dist[j] > max[1][0]) max[1] = ArrayUtils.addAll(new double[]{dist[j]}, inst.toDoubleArray());
                if (dist[j] < min[1][0]) min[1] = ArrayUtils.addAll(new double[]{dist[j]}, inst.toDoubleArray());
            }
            finalDist += distanceMetric.findDistance(p, q) * weight;
            this.mean[1] = 0.0f;
            for (int j = 0; j < nClasses; j++) {
                mean[1] += dist[j];
            }
            mean[1] /= nClasses;
            this.sd[1] = 0.0f;
            for (int j = 0; j < nClasses; j++) {
                sd[1] += Math.pow((dist[j] - mean[1]), 2);
            }
            sd[1] /= nClasses - 1;
            sd[1] = Math.sqrt(sd[1]);
        }
        return new double[]{this.findDistance((PriorModel)model1, (PriorModel)model2, sample), finalDist/(double)sample.getNInstances()};
    }

    public double[][] getMax(){
        return this.max;
    }

    public double[][] getMin(){
        return this.min;
    }

    public double[] getMean() {
        return this.mean;
    }

    public double[] getSd() {
        return this.sd;
    }

    private boolean vectorExists(double[] vector) {
        ArrayList<ArrayList<String>> attributesValues = this.sampler.getAllPossibleValues();
        boolean exists = true;
        for (int attributeIndex = 0; attributeIndex < vector.length; attributeIndex++) {
            exists = exists && attributesValues.get(attributeIndex).contains(Double.toString(vector[attributeIndex]));
        }
        return exists && this.combinationFreq[this.convertInstToHash(vector)][0] > 0.0f ;
    }

    private int convertInstToHash(double[] instance) {
        int hash = 0;
        for (int i = 0; i < instance.length; i++) {
            int current_hash = 1;
            for (int j = i - 1; j >= 0; j--) {
                current_hash *= this.sampler.getAllPossibleValues().get(j).size();
            }
            current_hash *= this.sampler.getAllPossibleValues().get(i).indexOf(Double.toString(instance[i]));
            hash += current_hash;
        }
        return hash;
    }

    private double[] convertHashToInst(int hash) {
        double[] instance = new double[this.sampler.getAllPossibleValues().size()];
        for (int i = instance.length - 1; i >= 0; i--) {
            int current_value = 1;
            for (int j = i - 1; j >= 0; j--) {
                current_value *= this.sampler.getAllPossibleValues().get(j).size();
            }
            current_value *= (int) instance[i];
            instance[i] = hash % current_value;
            hash = hash / current_value;
        }
        return instance;
    }

    private int convertClassToHash(double value) {
        return this.sampler.getAllClasses().indexOf(Double.toString(value));
    }
}
