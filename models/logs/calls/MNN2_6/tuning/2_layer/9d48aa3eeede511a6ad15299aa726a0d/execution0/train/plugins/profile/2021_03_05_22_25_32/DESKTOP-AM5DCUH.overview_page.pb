�	)�QG��3@)�QG��3@!)�QG��3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-)�QG��3@���A2@1�J>v(�?Ah�,{�?IY�oC��?*	     @G@2U
Iterator::Model::ParallelMapV2�ZӼ��?!�&�h��>@)�ZӼ��?1�&�h��>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!P?���O;@)�g��s��?1�.���6@:Preprocessing2F
Iterator::Model��A�f�?!y�GyF@)S�!�uq{?1�DM4�,@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ZӼ�?!���{�5@)S�!�uq{?1�DM4�,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!n��@)y�&1�l?1n��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip-C��6�?!�n��K@)��_vOf?1:�s�9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!B!�@)�J�4a?1B!�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIXu3M»X@Q�"�l�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���A2@���A2@!���A2@      ��!       "	�J>v(�?�J>v(�?!�J>v(�?*      ��!       2	h�,{�?h�,{�?!h�,{�?:	Y�oC��?Y�oC��?!Y�oC��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qXu3M»X@y�"�l�?�"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�k�X���?!�k�X���?"5
sequential/dense/MatMulMatMul�~7�=�?!*���v��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�rp��ģ?!�rp����?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�rp��ģ?!x��v8��?0"7
sequential/dense_1/MatMulMatMul�rp��ģ?!*���v��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulx��v8��?!�_�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�rp��ē?!8&D�n��?"!
Adam/PowPow�rp��ē?!d-�@���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�rp��ē?!�4R��?"7
sequential/dense_2/MatMulMatMul�rp��ē?!�;�]P�?0Q      Y@Y7��Moz2@a���,daT@q"�D沄X@y�Y-���?"�
both�Your program is POTENTIALLY input-bound because 91.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 