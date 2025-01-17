�	�tp��6@�tp��6@!�tp��6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�tp��6@6�EaA4@1Di��?Ab��4�8�?I��}�u��?*	33333�J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateǺ���?!�S{�D@)�0�*�?1;��C@:Preprocessing2U
Iterator::Model::ParallelMapV2� �	��?!$I�$I�<@)� �	��?1$I�$I�<@:Preprocessing2F
Iterator::Model��_vO�?!H�4�	D@)a��+ey?1�N��l'@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ���v?!�S{�$@)���_vOn?1��dDPu@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipaTR'���?!�G�<��M@)�~j�t�h?14鏃qC@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!R��K3@)ŏ1w-!_?1R��K3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap0*��D�?!��K3�E@)��_�LU?1�R��K@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!��B��?)����MbP?1��B��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice-C��6J?!|��h��?)-C��6J?1|��h��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIЍ��X@Q�9@���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	6�EaA4@6�EaA4@!6�EaA4@      ��!       "	Di��?Di��?!Di��?*      ��!       2	b��4�8�?b��4�8�?!b��4�8�?:	��}�u��?��}�u��?!��}�u��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qЍ��X@y�9@���?�"5
sequential/dense/MatMulMatMul-$�o��?!-$�o��?0"C
%gradient_tape/sequential/dense/MatMulMatMul���P�?!�W|-G��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch5�Gx��?!�8���?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul5�Gx��?!��t�C�?"7
sequential/dense_1/MatMulMatMul��E���?!$�xg_e�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�69`|��?!���.��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�69`|��?!�����?"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrado�v����?!���?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam��_L؎?!�����?"
Sum_7Sum��_L؎?!6,|%��?Q      Y@Y{	�%��1@a�����T@qё�N`yX@y��,+Q�?"�
both�Your program is POTENTIALLY input-bound because 89.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 