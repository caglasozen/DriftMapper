package main.models.frequency;

import main.analyse.ExperimentResult;
import main.models.Model;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by loongkuan on 28/11/2016.
 **/

public abstract class BaseFrequencyModel extends Model {

    protected int[] hashBases;

    public abstract void changeAttributeSubsetLength(int length);

    protected abstract int findFv(Instance instance, int[] attributesSubset);
    protected abstract int findFy(int classIndex);
    protected abstract int findFvy(Instance instance, int[] attributesSubset, int classIndex);
    protected abstract Set<Integer> getAllHashes(int[] attributeSubset);

    protected void setDataset(Instances dataset) {
        this.allInstances = new Instances(dataset, dataset.size());
        // Get bases for hash
        int base = 1;
        this.hashBases = new int[dataset.numAttributes() - 1];
        for (int i = 0; i < dataset.numAttributes() - 1; i++) {
            this.hashBases[i] = base;
            base *= dataset.attribute(attributesAvailable[i]).numValues();
        }
        this.reset();
    }

    protected int instanceToPartialHash(Instance instance, int[] attributeSubset) {
        int hash = 0;
        // TODO: Don't rely on class being last attribute
        // Iterate over attributes in strippedInstance
        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            // If not class or active attribute set missing
            if (ArrayUtils.contains(attributeSubset, i)) {
                hash += this.hashBases[i] * (int)instance.value(i);
            }
        }
        return hash;
    }

    protected Instance partialHashToInstance(int partialHash, int[] activeAttributes) {
        Instance partialInstance = new DenseInstance(this.allInstances.numAttributes());
        partialInstance.setDataset(this.allInstances);

        for (int i = partialInstance.numAttributes() - 2; i >= 0; i--) {
            if (ArrayUtils.contains(activeAttributes, i)) {
                partialInstance.setValue(i, partialHash / this.hashBases[i]);
                partialHash = partialHash % this.hashBases[i];
            }
            else {
                partialInstance.setMissing(i);
            }
        }
        partialInstance.setClassMissing();
        return partialInstance;
    }

    protected boolean isPartialHashEqualHash(int partialHash, int hash, int[] attributeSubset) {
        int rem = hash - partialHash;
        int quo;
        for (int i = this.allInstances.numAttributes() - 2; i >= 0; i--) {
            if (rem < 0) return false;
            if (!ArrayUtils.contains(attributeSubset, i)) {
                quo = rem / this.hashBases[i];
                if (quo >= this.allInstances.attribute(i).numValues()) return false;
                rem = rem % this.hashBases[i];
            }
        }
        return rem == 0;
    }

    protected boolean validAttributeSubset(int[] attributeSubset) {
        boolean valid = true;
        for (int attributeIndex : attributeSubset) {
            valid = valid && ArrayUtils.contains(this.attributesAvailable, attributeIndex);
        }
        return valid && attributeSubset.length == this.attributeSubsetLength;
    }

    protected static ArrayList<Integer> sampleHashes(ArrayList<Integer> hashes, double sampleScale) {
        ArrayList<Integer> sampledHashes = new ArrayList<>();
        ArrayList<Integer> allHash = new ArrayList<>(hashes);
        Collections.shuffle(allHash);
        for (int i = 0; i < Math.min((int)(hashes.size() / sampleScale), hashes.size()); i++) {
            sampledHashes.add(allHash.get(i));
        }
        return sampledHashes;
    }

    private ArrayList<Integer> mergeHashes(BaseFrequencyModel model, int[] attributeSubset) {
        Set<Integer> hashes = new HashSet<>();
        hashes.addAll(this.getAllHashes(attributeSubset));
        hashes.addAll(model.getAllHashes(attributeSubset));
        return new ArrayList<>(hashes);
    }

    // TODO: Try and separate these into smaller functions
    @Override
    public ExperimentResult findCovariateDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<Integer> allHashes = mergeHashes(modelAfter, attributeSubset);
        allHashes = sampleHashes(allHashes, sampleScale);

        double[] p = new double[allHashes.size()];
        double[] q = new double[allHashes.size()];
        double[] separateDistance = new double[allHashes.size()];
        double[][] instanceValues = new double[allHashes.size()][this.allInstances.numAttributes()];
        for (int i = 0; i < allHashes.size(); i++) {
            Instance instance = this.partialHashToInstance(allHashes.get(i), attributeSubset);
            p[i] = this.allInstances.size() == 0 ?
                    0 : (double)this.findFv(instance, attributeSubset) / (double)this.allInstances.size();
            q[i] = modelAfter.allInstances.size() == 0 ?
                    0 : (double)modelAfter.findFv(instance, attributeSubset) / (double)modelAfter.allInstances.size();
            separateDistance[i] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
            instanceValues[i] = instance.toDoubleArray();
        }
        double finalDistance = this.distanceMetric.findDistance(p, q) * sampleScale;
        ExperimentResult finalResult = new ExperimentResult(finalDistance, separateDistance, instanceValues);
        return finalResult;
    }

    @Override
    public ExperimentResult findJointDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<Integer> allHashes = mergeHashes(modelAfter, attributeSubset);

        int nClass =  this.allInstances.numClasses();
        double[] p = new double[allHashes.size() * nClass];
        double[] q = new double[allHashes.size() * nClass];
        double[] separateDistance = new double[allHashes.size() * nClass];
        double[][] instanceValues = new double[allHashes.size() * nClass][this.allInstances.numAttributes()];
        for (int i = 0; i < allHashes.size(); i++) {
            Instance instance = this.partialHashToInstance(allHashes.get(i), attributeSubset);
            for(int classIndex = 0; classIndex < nClass; classIndex++) {
                p[i*nClass + classIndex] = this.allInstances.size() == 0 ?
                        0 : (double)this.findFvy(instance, attributeSubset, classIndex) / (double)this.allInstances.size();
                q[i*nClass + classIndex] = modelAfter.allInstances.size() == 0 ?
                        0 : (double) modelAfter.findFvy(instance, attributeSubset, classIndex) / (double)modelAfter.allInstances.size();
                separateDistance[i*nClass + classIndex] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
                instanceValues[i*nClass + classIndex] = instance.toDoubleArray();
            }
        }
        double finalDistance = this.distanceMetric.findDistance(p, q) * sampleScale;
        ExperimentResult finalResult = new ExperimentResult(finalDistance, separateDistance, instanceValues);
        return finalResult;
    }

    @Override
    public ExperimentResult findLikelihoodDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<Integer> allHashes = mergeHashes(modelAfter, attributeSubset);

        double[] p = new double[allHashes.size()];
        double[] q = new double[allHashes.size()];
        double[] separateDistance = new double[allHashes.size()];
        double[][] instanceValues = new double[allHashes.size()][this.allInstances.numAttributes()];
        double finalDistance = 0.0f;
        for (int classIndex = 0; classIndex < this.allInstances.numClasses(); classIndex++) {
            for (int i = 0; i < allHashes.size(); i++) {
                Instance instance = this.partialHashToInstance(allHashes.get(i), attributeSubset);
                instance.setClassValue(classIndex);
                p[i] = this.findFy(classIndex) == 0 ?
                        0 : (double) this.findFvy(instance, attributeSubset, classIndex) / (double) this.findFy(classIndex);
                q[i] = modelAfter.findFy(classIndex) == 0 ?
                        0 : (double) modelAfter.findFvy(instance, attributeSubset, classIndex) / (double) modelAfter.findFy(classIndex);
                separateDistance[i] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
                instanceValues[i] = instance.toDoubleArray();
            }
            finalDistance += this.distanceMetric.findDistance(p, q) * sampleScale * (
                    ((double)this.findFy(classIndex) / (double)this.allInstances.size()) +
                            ((double)modelAfter.findFy(classIndex) / (double)modelAfter.allInstances.size())
            )/2;
        }
        ExperimentResult finalResult = new ExperimentResult(finalDistance, separateDistance, instanceValues);
        return finalResult;
    }

    @Override
    public ExperimentResult findPosteriorDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<Integer> allHashes = mergeHashes(modelAfter, attributeSubset);

        double[] p = new double[this.allInstances.numClasses()];
        double[] q = new double[this.allInstances.numClasses()];
        double[] separateDistance = new double[this.allInstances.numClasses()];
        double[][] instanceValues = new double[this.allInstances.numClasses()][this.allInstances.numAttributes()];
        double finalDistance = 0.0f;
        for (int i = 0; i < allHashes.size(); i++) {
            Instance instance = this.partialHashToInstance(allHashes.get(i), attributeSubset);
            for (int classIndex = 0; classIndex < this.allInstances.numClasses(); classIndex++) {
                instance.setClassValue(classIndex);
                p[classIndex] = this.findFv(instance, attributeSubset) == 0 ?
                        0 : (double) this.findFvy(instance, attributeSubset, classIndex) / (double) this.findFv(instance, attributeSubset);
                q[classIndex] = modelAfter.findFv(instance, attributeSubset) == 0 ?
                        0 : (double) modelAfter.findFvy(instance, attributeSubset, classIndex) / (double) modelAfter.findFv(instance, attributeSubset);
                separateDistance[classIndex] = this.distanceMetric.findDistance(new double[]{p[classIndex]}, new double[]{q[classIndex]});
                instanceValues[classIndex] = instance.toDoubleArray();
            }
            finalDistance += this.distanceMetric.findDistance(p, q) * sampleScale * (
                    ((double)this.findFv(instance, attributeSubset)/(double)this.allInstances.size()) +
                            ((double)modelAfter.findFv(instance, attributeSubset)/(double)modelAfter.allInstances.size())
            ) /2;
        }
        ExperimentResult finalResult = new ExperimentResult(finalDistance, separateDistance, instanceValues);
        return finalResult;
    }

    @Override
    public double peakCovariateDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        return this.findCovariateDistance(modelToCompare, attributeSubset, sampleScale).actualResult;
    }

    @Override
    public double peakJointDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        return this.findJointDistance(modelToCompare, attributeSubset, sampleScale).actualResult;
    }

    @Override
    public double peakLikelihoodDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        return this.findLikelihoodDistance(modelToCompare, attributeSubset, sampleScale).actualResult;
    }

    @Override
    public double peakPosteriorDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        return this.findPosteriorDistance(modelToCompare, attributeSubset, sampleScale).actualResult;
    }
}
